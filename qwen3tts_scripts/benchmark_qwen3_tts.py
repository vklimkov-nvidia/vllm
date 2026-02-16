"""Benchmark script for Qwen3 TTS model on vLLM.

Usage:
    python benchmarks/benchmark_qwen3_tts.py \
        --model dummy_qwen3_tts_model \
        --concurrency 16 \
        --num-requests 512 \
        --input-len 128 \
        --output-len 256
"""

import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

import argparse
import asyncio
import logging
import time
import uuid
from typing import Any, Dict

import numpy as np
import torch

# Suppress verbose vLLM logging
logging.getLogger("vllm").setLevel(logging.WARNING)

try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
except ImportError:
    print("Error: Failed to import vllm.")
    print("Please install vllm: pip install vllm")
    exit(1)


async def run_request(
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    output_steps: int,
    hidden_size: int,
    metrics: Dict[str, Any],
    request_id: str,
):
    """
    Sends a single Qwen3 TTS request to the vLLM engine and records metrics.

    The model uses combined_embeddings as custom input.  Each autoregressive
    step produces codec tokens (16 groups) plus next_input_embeddings which
    are fed back via append_request.
    """

    # Random prefill embeddings – shape [input_num_tokens, hidden_size]
    prefill_emb = torch.randn(
        input_num_tokens, hidden_size, dtype=torch.bfloat16
    )

    inputs = {
        "prompt_token_ids": [0] * input_num_tokens,
        "custom_inputs": {
            "combined_embeddings": prefill_emb,
        },
    }

    request_start_time = time.perf_counter()
    last_token_time = None
    step_idx = 0

    try:
        results_generator = engine.generate(
            inputs, sampling_params=sampling_params, request_id=request_id
        )

        async for output in results_generator:
            now = time.perf_counter()

            custom_out = output.outputs[0].custom_outputs
            codec_tokens = custom_out["codes"][-1:]  # [1, 16]
            next_input = custom_out["next_input_embeddings"]

            # Track first 6 steps separately
            if step_idx < 6:
                token_latency = (
                    now - last_token_time
                    if last_token_time
                    else now - request_start_time
                )
                metrics["first_tokens"][step_idx].append(token_latency)
            elif last_token_time is not None:
                itl = now - last_token_time
                metrics["inter_token_latencies"].append(itl)

            last_token_time = now
            step_idx += 1

            # Fixed number of decode steps
            if step_idx >= output_steps:
                await engine.abort(request_id)
                break

            if output.finished:
                break

            # Feed next decode step embeddings
            new_custom_inputs = {
                "combined_embeddings": next_input[-1:, :],
            }
            await engine.append_request(
                request_id=request_id, custom_inputs=new_custom_inputs
            )

        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time

        metrics["request_latencies"].append(request_latency)
        metrics["completed_sequences"] += 1
        metrics["total_tokens"] += step_idx

    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        import traceback
        traceback.print_exc()
        metrics["failed_sequences"] += 1


def calculate_and_print_metrics(
    metrics: Dict[str, Any], total_time: float, args: argparse.Namespace
):
    """
    Calculates and prints the final benchmark statistics.
    """
    total_sequences = metrics["completed_sequences"]
    if total_sequences == 0:
        print("Error: No sequences completed.")
        return

    print(f"\n{'='*60}")
    print(f"Benchmark Results  (concurrency={args.concurrency}, "
          f"requests={args.num_requests}, "
          f"in={args.input_len}, out={args.output_len})")
    print(f"{'='*60}")

    # First 6 tokens (0-5) – mean latency for each position
    print("\n--- First 6 Tokens (mean latency in ms) ---")
    for i in range(6):
        if metrics["first_tokens"][i]:
            avg_ms = np.mean(metrics["first_tokens"][i]) * 1000
            p95_ms = np.percentile(metrics["first_tokens"][i], 95) * 1000
            print(f"  Token {i}: {avg_ms:.2f} ms (P95: {p95_ms:.2f} ms)")

    # Rest of tokens (6+) ITL
    if metrics["inter_token_latencies"]:
        avg_itl_ms = np.mean(metrics["inter_token_latencies"]) * 1000
        p95_itl_ms = np.percentile(metrics["inter_token_latencies"], 95) * 1000
        print(f"\n--- Tokens 6+ ITL ---")
        print(f"  Average: {avg_itl_ms:.2f} ms (P95: {p95_itl_ms:.2f} ms)")

    # Average total time per request
    avg_request_time = np.mean(metrics["request_latencies"])
    p95_request_time = np.percentile(metrics["request_latencies"], 95)
    print(f"\n--- Request Latency ---")
    print(f"  Average: {avg_request_time:.2f} s (P95: {p95_request_time:.2f} s)")

    # Throughput
    total_tokens = metrics["total_tokens"]
    print(f"\n--- Throughput ---")
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Completed sequences: {total_sequences}")
    print(f"  Failed sequences: {metrics['failed_sequences']}")
    print(f"  Total decode steps: {total_tokens}")
    print(f"  Throughput: {total_tokens / total_time:.2f} steps/s")
    print(f"  Sequence throughput: {total_sequences / total_time:.2f} seq/s")
    print(f"{'='*60}\n")


async def worker(
    worker_id: int,
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    output_steps: int,
    hidden_size: int,
    metrics: Dict[str, Any],
):
    """
    A persistent worker that continuously sends requests until
    the global request counter reaches zero.
    """
    while True:
        async with metrics["lock"]:
            if metrics["requests_to_run"] <= 0:
                break
            metrics["requests_to_run"] -= 1

        request_id = f"benchmark-w{worker_id}-{uuid.uuid4()}"
        await run_request(
            engine,
            sampling_params,
            input_num_tokens,
            output_steps,
            hidden_size,
            metrics,
            request_id,
        )


def init_metrics(num_requests: int):
    return {
        "request_latencies": [],
        "inter_token_latencies": [],
        "first_tokens": [[] for _ in range(6)],
        "completed_sequences": 0,
        "failed_sequences": 0,
        "total_tokens": 0,
        "requests_to_run": num_requests,
        "lock": asyncio.Lock(),
    }


async def main():
    parser = argparse.ArgumentParser(
        description="vLLM Qwen3 TTS Benchmarking Script"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dummy_qwen3_tts_model",
        help="Model name or path (default: dummy_qwen3_tts_model)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "-m",
        "--num-requests",
        type=int,
        default=512,
        help="Total number of requests to send",
    )
    parser.add_argument(
        "-i",
        "--input-len",
        type=int,
        default=128,
        help="Prefix / prompt length (number of input tokens)",
    )
    parser.add_argument(
        "-o",
        "--output-len",
        type=int,
        default=256,
        help="Number of decode steps to run per request",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model length (context size)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.7,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
        help="Hidden size for the combined_embeddings input (must match model config)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture and run in eager mode",
    )

    args = parser.parse_args()

    print(
        f"Benchmark: concurrency={args.concurrency}, "
        f"requests={args.num_requests}, "
        f"in={args.input_len}, out={args.output_len}"
    )

    # Compute max_num_batched_tokens to accommodate concurrency
    max_model_len = max(args.max_model_len, args.input_len + args.output_len)
    max_num_batched_tokens = max_model_len * args.concurrency * 2

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem,
        skip_tokenizer_init=True,
        load_format="dummy",
        disable_log_stats=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        compilation_config={"cudagraph_mode": "PIECEWISE"},
    )

    print("Initializing engine...")
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=max_model_len,
        skip_sampling=True,
    )

    # --- Warmup + Benchmark ---
    for run, num_requests in enumerate(
        [3 * args.concurrency, args.num_requests]
    ):
        metrics = init_metrics(num_requests)

        if run == 0:
            print("Warmup...")
        else:
            print("Running benchmark...")

        start_time = time.perf_counter()

        tasks = [
            asyncio.create_task(
                worker(
                    worker_id=i,
                    engine=engine,
                    sampling_params=sampling_params,
                    input_num_tokens=args.input_len,
                    output_steps=args.output_len,
                    hidden_size=args.hidden_size,
                    metrics=metrics,
                )
            )
            for i in range(args.concurrency)
        ]

        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if run == 0:
            print(f"Warmup done in {total_time:.2f}s "
                  f"({metrics['completed_sequences']} sequences, "
                  f"{metrics['failed_sequences']} failed)")
        else:
            calculate_and_print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")

