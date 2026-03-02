import argparse
import asyncio
import time
import uuid
import logging
from typing import Dict, Any
import numpy as np
import torch

logging.getLogger("vllm").setLevel(logging.WARNING)

try:
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
except ImportError:
    print("Error: Failed to import vllm.")
    print("Please install vllm: pip install vllm")
    exit(1)


async def run_request(
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    hidden_size: int,
    metrics: Dict[str, Any],
    request_id: str,
):
    """
    Sends a single request to the vLLM engine and records metrics.
    Prefills with random combined embeddings, then decodes step-by-step.
    """
    combined_embeds = torch.randn(
        input_num_tokens, hidden_size, dtype=torch.bfloat16
    )

    inputs = {
        "prompt_token_ids": [0] * input_num_tokens,
        "custom_inputs": {"combined_embeds": combined_embeds},
    }

    request_start_time = time.perf_counter()
    last_token_time = None
    token_idx = 0

    try:
        async for output in engine.generate(
            inputs, sampling_params=sampling_params, request_id=request_id
        ):
            now = time.perf_counter()

            if token_idx < 6:
                token_latency = (
                    now - last_token_time
                    if last_token_time
                    else now - request_start_time
                )
                metrics["first_tokens"][token_idx].append(token_latency)
            elif last_token_time is not None:
                itl = now - last_token_time
                metrics["inter_token_latencies"].append(itl)

            last_token_time = now
            token_idx += 1

            if output.finished:
                break

            step_embeds = torch.randn(1, hidden_size, dtype=torch.bfloat16)
            await engine.append_request(
                request_id=request_id,
                custom_inputs={"combined_embeds": step_embeds},
            )

        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time

        metrics["request_latencies"].append(request_latency)
        metrics["completed_sequences"] += 1
        metrics["total_tokens"] += token_idx

    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        import traceback

        traceback.print_exc()
        metrics["failed_sequences"] += 1


def calculate_and_print_metrics(
    metrics: Dict[str, Any], total_time: float, args: argparse.Namespace
):
    total_sequences = metrics["completed_sequences"]
    if total_sequences == 0:
        print("Error: No sequences completed.")
        return

    total_tokens = metrics["total_tokens"]
    avg_tokens_per_sec = total_tokens / total_time
    avg_seq_per_sec = total_sequences / total_time

    print("\n--- Nemotron LLM Benchmark Results ---")
    print(
        f"Concurrency: {args.concurrency}, "
        f"Input: {args.input_len}, Output: {args.output_len}, "
        f"Hidden: {args.hidden_size}"
    )
    print(f"Total duration: {total_time:.2f} s")
    print(f"Completed: {total_sequences}, Failed: {metrics['failed_sequences']}")

    print("\n--- First 6 Tokens (mean latency in ms) ---")
    for i in range(6):
        if metrics["first_tokens"][i]:
            avg_ms = np.mean(metrics["first_tokens"][i]) * 1000
            p95_ms = np.percentile(metrics["first_tokens"][i], 95) * 1000
            label = "TTFT (prefill)" if i == 0 else f"Token {i}"
            print(f"  {label}: {avg_ms:.2f} ms (P95: {p95_ms:.2f} ms)")

    if metrics["inter_token_latencies"]:
        avg_itl_ms = np.mean(metrics["inter_token_latencies"]) * 1000
        p50_itl_ms = np.percentile(metrics["inter_token_latencies"], 50) * 1000
        p95_itl_ms = np.percentile(metrics["inter_token_latencies"], 95) * 1000
        p99_itl_ms = np.percentile(metrics["inter_token_latencies"], 99) * 1000
        print(f"\n--- Tokens 6+ ITL ---")
        print(
            f"  Mean: {avg_itl_ms:.2f} ms, "
            f"P50: {p50_itl_ms:.2f} ms, "
            f"P95: {p95_itl_ms:.2f} ms, "
            f"P99: {p99_itl_ms:.2f} ms"
        )

    avg_request_time = np.mean(metrics["request_latencies"])
    p95_request_time = np.percentile(metrics["request_latencies"], 95)
    print(f"\n--- Request Latency ---")
    print(f"  Average: {avg_request_time:.2f} s (P95: {p95_request_time:.2f} s)")

    print(f"\n--- Throughput ---")
    print(f"  Sequences/sec: {avg_seq_per_sec:.2f}")
    print(f"  Tokens/sec: {avg_tokens_per_sec:.2f}")

    # S2S real-time factor: each LLM step corresponds to 80ms of audio
    audio_duration_per_seq = args.output_len * 0.08
    avg_rtf = avg_request_time / audio_duration_per_seq
    print(f"\n--- S2S Real-Time Factor (80ms/frame) ---")
    print(f"  RTF: {avg_rtf:.4f}x (< 1.0 means faster than real-time)")
    print("---------------------------------------")


async def worker(
    worker_id: int,
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    hidden_size: int,
    metrics: Dict[str, Any],
):
    while True:
        async with metrics["lock"]:
            if metrics["requests_to_run"] <= 0:
                break
            metrics["requests_to_run"] -= 1

        request_id = f"benchmark-w{worker_id}-{uuid.uuid4()}"
        await run_request(
            engine, sampling_params, input_num_tokens, hidden_size, metrics, request_id
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
        description="vLLM Nemotron LLM (S2S backbone) Benchmarking Script"
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
        help="Prefill prompt length (number of combined embeddings)",
    )
    parser.add_argument(
        "-o",
        "--output-len",
        type=int,
        default=256,
        help="Number of decode steps (one per 80ms audio frame in S2S)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4480,
        help="Model hidden size for generating random embeddings",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Maximum model length (context size)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.7,
        help="GPU memory utilization (0.0 to 1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the converted Nemotron LLM model directory",
    )
    parser.add_argument(
        "--load-format",
        type=str,
        default="auto",
        choices=["auto", "dummy"],
        help="Load format: 'auto' for real weights, 'dummy' for random weights",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graphs)",
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
        help="Run with torch profiler",
    )
    args = parser.parse_args()

    print("Starting Nemotron LLM benchmark...")
    print(
        f"Concurrency: {args.concurrency}, Requests: {args.num_requests}, "
        f"Input: {args.input_len}, Output: {args.output_len}, "
        f"Hidden: {args.hidden_size}"
    )

    engine_args_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "trust_remote_code": True,
        "mamba_ssm_cache_dtype": "float32",
        "skip_tokenizer_init": True,
        "enable_prefix_caching": False,
        "enforce_eager": args.enforce_eager,
        "disable_log_stats": True,
        "input_coalesce_timeout_ms": args.input_coalesce_timeout_ms,
    }
    if args.load_format == "dummy":
        engine_args_kwargs["load_format"] = "dummy"

    engine_args = AsyncEngineArgs(**engine_args_kwargs)
    engine = AsyncLLM.from_engine_args(engine_args)

    if args.profile:
        await engine.start_profile()

    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        skip_sampling=True,
        ignore_eos=True,
    )

    warmup_num = 0 if args.no_warmup else 3 * args.concurrency
    for run, num_requests in enumerate([warmup_num, args.num_requests]):
        if num_requests == 0:
            continue
        metrics = init_metrics(num_requests)

        run_name = "Warmup" if run == 0 else "Benchmark"
        print(f"\n--- Starting {run_name} ({num_requests} requests) ---")
        start_time = time.perf_counter()

        tasks = []
        for i in range(args.concurrency):
            tasks.append(
                asyncio.create_task(
                    worker(
                        worker_id=i,
                        engine=engine,
                        sampling_params=sampling_params,
                        input_num_tokens=args.input_len,
                        hidden_size=args.hidden_size,
                        metrics=metrics,
                    )
                )
            )

        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f"{run_name} finished in {total_time:.2f}s")

        if run > 0:
            if args.profile:
                await engine.stop_profile()
            calculate_and_print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
