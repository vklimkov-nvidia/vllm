"""Benchmark script for Qwen3 TTS model on vLLM.

Usage:
    python benchmarks/benchmark_qwen3_tts.py \
        --model dummy_qwen3_tts_model \
        --concurrency 16 \
        --num-requests 512 \
        --input-len 128 \
        --output-len 256

    # With shared-memory decode channel:
    python benchmarks/benchmark_qwen3_tts.py \
        --model dummy_qwen3_tts_model \
        --concurrency 16 \
        --use-shm

    # With torch profiler (output dir via VLLM_TORCH_PROFILER_DIR):
    #   VLLM_TORCH_PROFILER_DIR=/path/to/traces python benchmark_qwen3_tts.py --profile
    python benchmarks/benchmark_qwen3_tts.py \
        --model dummy_qwen3_tts_model \
        --profile

Scheduling note (why you see 1+3 or 2+2 prefills instead of 4):
    add_request messages were already in the queue when the engine woke.
    With 4 concurrent workers each calling add_request, that number is a race
    between the client event loop and the engine's receiver (ZMQ/shm), so you
    often see 1 or 2 prefills in the first step, then 3 or 2 in the next, and
    then prefill+decode mixed as the first request(s) move to decode while
    the rest are still prefill. Use --burst-first-batch to add the first
    concurrency requests from a single task so they are all enqueued before
    the first step (deterministic batch prefill).
"""

import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

import argparse
import asyncio
import logging
import random
import time
import uuid
from typing import Any, Dict

import numpy as np
import torch

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
        queue = await engine.add_request(request_id, inputs, sampling_params)

        while True:
            output = await queue.get()
            now = time.perf_counter()

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
            next_input = output.outputs[0].custom_outputs["next_input_embeddings"]
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


def _decode_loop_sync(
    engine: AsyncLLM,
    output_steps: int,
    metrics: Dict[str, Any],
    request_id: str,
    prefill_custom_outputs: Dict[str, Any],
    prefill_token_time: float,
    request_start_time: float,
):
    """Run the entire decode loop in a plain OS thread (no asyncio).

    After prefill completes on the event loop, this function takes over.
    Each step is: prepare input from previous output -> shm decode_step
    (futex write+wait) -> record metrics.  No event-loop round-trips
    means zero queuing delay between concurrent requests.
    """
    next_input = prefill_custom_outputs["next_input_embeddings"]
    last_token_time = prefill_token_time
    token_idx = 1

    try:
        for _ in range(output_steps - 1):
            custom_inputs = {"combined_embeddings": next_input[-1:, :]}
            custom_outputs = engine.decode_step_shm(
                request_id, custom_inputs=custom_inputs,
            )
            now = time.perf_counter()

            next_input = custom_outputs["next_input_embeddings"]

            if token_idx < 6:
                metrics["first_tokens"][token_idx].append(
                    now - last_token_time
                )
            else:
                metrics["inter_token_latencies"].append(
                    now - last_token_time
                )

            last_token_time = now
            token_idx += 1

        request_end_time = time.perf_counter()
        metrics["request_latencies"].append(
            request_end_time - request_start_time
        )
        metrics["completed_sequences"] += 1
        metrics["total_tokens"] += token_idx

    except Exception as e:
        print(f"Request {request_id} decode loop failed: {e}")
        import traceback
        traceback.print_exc()
        metrics["failed_sequences"] += 1


async def run_request_shm(
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    output_steps: int,
    hidden_size: int,
    metrics: Dict[str, Any],
    request_id: str,
):
    """
    SHM mode: prefill via ZMQ (add_request on event loop), then the
    entire decode loop runs in a dedicated OS thread — no event-loop
    round-trips between steps.
    """
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

    try:
        queue = await engine.add_request(request_id, inputs, sampling_params)
        prefill_output = await queue.get()
        prefill_token_time = time.perf_counter()
        metrics["first_tokens"][0].append(
            prefill_token_time - request_start_time
        )

        custom_out = prefill_output.outputs[0].custom_outputs

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _decode_loop_sync,
            engine,
            output_steps,
            metrics,
            request_id,
            custom_out,
            prefill_token_time,
            request_start_time,
        )

        await engine.abort(request_id)

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


_JITTER_PCT = 20

def _jittered_len(base: int) -> int:
    """Return *base* perturbed by uniform +/- _JITTER_PCT %, min 1."""
    factor = 1.0 + random.uniform(-_JITTER_PCT, _JITTER_PCT) / 100.0
    return max(1, int(round(base * factor)))


async def worker(
    worker_id: int,
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    output_steps: int,
    hidden_size: int,
    metrics: Dict[str, Any],
    use_shm: bool = False,
    len_jitter: bool = False,
    randomize_delay: bool = False,
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

        if randomize_delay:
            tokens = (_JITTER_PCT / 100) * output_steps
            sec = tokens / 12.5
            await asyncio.sleep(random.uniform(0, sec))

        req_input_len = _jittered_len(input_num_tokens) if len_jitter else input_num_tokens
        req_output_len = _jittered_len(output_steps) if len_jitter else output_steps

        request_id = f"benchmark-w{worker_id}-{uuid.uuid4()}"
        if use_shm:
            await run_request_shm(
                engine,
                sampling_params,
                req_input_len,
                req_output_len,
                hidden_size,
                metrics,
                request_id,
            )
        else:
            await run_request(
                engine,
                sampling_params,
                req_input_len,
                req_output_len,
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
    parser.add_argument(
        "--use-shm",
        action="store_true",
        help="Use shared-memory decode channel instead of ZMQ for "
             "decode-step I/O (requires custom_output_specs in model config)",
    )
    parser.add_argument(
        "--input-coalesce-timeout-ms",
        type=float,
        default=0,
        help="Wait up to this many ms for all requests to receive custom "
             "inputs before running a forward pass (0 to disable)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup run",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Call engine.start_profile() before and engine.stop_profile() after "
             "the benchmark run. Trace output dir: VLLM_TORCH_PROFILER_DIR.",
    )
    parser.add_argument(
        "--randomize-len",
        action="store_true",
        help="Randomize input/output lengths per request by +/- 15%%",
    )
    parser.add_argument(
        "--randomize-delay",
        action="store_true",
        help="Add a random delay between requests per worker.  Delay is "
             "uniform(0, T) where T = (_JITTER_PCT/100)*output_len / 12.5 s.",
    )

    args = parser.parse_args()

    mode = "SHM" if args.use_shm else "ZMQ"
    jitter_str = f", jitter=±{_JITTER_PCT}%" if args.randomize_len else ""
    delay_str = ", randomize-delay" if args.randomize_delay else ""
    print(
        f"Benchmark ({mode}): concurrency={args.concurrency}, "
        f"requests={args.num_requests}, "
        f"in={args.input_len}, out={args.output_len}{jitter_str}{delay_str}"
        + (" [profiling enabled]" if args.profile else "")
    )

    jitter_headroom = (1.0 + _JITTER_PCT / 100.0) if args.randomize_len else 1.0
    max_input = int(args.input_len * jitter_headroom) + 1
    max_output = int(args.output_len * jitter_headroom) + 1
    max_model_len = max(args.max_model_len, max_input + max_output)
    #max_num_batched_tokens = args.concurrency * 32

    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="bfloat16",
        max_model_len=max_model_len,
        #max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_mem,
        skip_tokenizer_init=True,
        load_format="dummy",
        disable_log_stats=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        compilation_config={"cudagraph_mode": "PIECEWISE"},
        shm_decode=args.use_shm,
        input_coalesce_timeout_ms=args.input_coalesce_timeout_ms,
        #enable_chunked_prefill=True,
    )

    print("Initializing engine...")
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=max_model_len,
        skip_sampling=True,
    )

    # --- Warmup + Benchmark ---
    warmup_num = 0 if args.no_warmup else 3 * args.concurrency
    for run, num_requests in enumerate(
        [warmup_num, args.num_requests]
    ):
        if num_requests == 0:
            continue
        metrics = init_metrics(num_requests)

        run_name = "Warmup" if run == 0 else "Benchmark"
        print(f"\n--- Starting {run_name} ({num_requests} requests) ---")

        start_time = time.perf_counter()

        if run > 0 and args.profile:
            await engine.start_profile()
        try:
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
                        use_shm=args.use_shm,
                        len_jitter=args.randomize_len,
                        randomize_delay=args.randomize_delay,
                    )
                )
                for i in range(args.concurrency)
            ]
            await asyncio.gather(*tasks)
        finally:
            if run > 0 and args.profile:
                await engine.stop_profile()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"{run_name} finished in {total_time:.2f}s "
              f"({metrics['completed_sequences']} sequences, "
              f"{metrics['failed_sequences']} failed)")

        if run > 0:
            calculate_and_print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")

