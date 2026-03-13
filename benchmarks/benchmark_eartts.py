import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

import argparse
import asyncio
import concurrent.futures
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
    metrics: Dict[str, Any],
    request_id: str,
):
    """
    ZMQ mode: prefills with random acoustic/text inputs, then decodes
    step-by-step via add_request/queue + append_request.
    """
    model_dtype = engine.vllm_config.model_config.dtype
    prompt_acoustic_tokens = torch.randint(
        0, 1024, (input_num_tokens, 31), dtype=torch.int32
    )
    bos_mask = torch.zeros(input_num_tokens, dtype=model_dtype)
    bos_mask[0] = 1.0

    inputs = {
        "prompt_token_ids": [0] * input_num_tokens,
        "custom_inputs": {
            "acoustic_tokens": prompt_acoustic_tokens,
            "text_tokens": torch.zeros(input_num_tokens, dtype=torch.int32),
            "text_mask": torch.zeros(input_num_tokens, dtype=model_dtype),
            "bos_mask": bos_mask,
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

            acoustic_tokens = output.outputs[0].custom_outputs.get(
                "acoustic_tokens"
            )

            if step_idx < 6:
                token_latency = (
                    now - last_token_time
                    if last_token_time
                    else now - request_start_time
                )
                metrics["first_tokens"][step_idx].append(token_latency)
            elif last_token_time is not None:
                metrics["inter_token_latencies"].append(now - last_token_time)

            last_token_time = now
            step_idx += 1

            if step_idx >= output_steps:
                await engine.abort(request_id)
                break

            step_acoustic_tokens = acoustic_tokens[-1:]
            current_text_token = torch.randint(
                0, 10000, (1,), dtype=torch.int32
            )
            await engine.append_request(
                request_id=request_id,
                custom_inputs={
                    "acoustic_tokens": step_acoustic_tokens,
                    "text_tokens": current_text_token,
                    "text_mask": torch.ones(1, dtype=model_dtype),
                    "bos_mask": torch.zeros(1, dtype=model_dtype),
                },
            )

        request_end_time = time.perf_counter()
        metrics["request_latencies"].append(
            request_end_time - request_start_time
        )
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
    model_dtype: torch.dtype,
    metrics: Dict[str, Any],
    request_id: str,
    prefill_acoustic_tokens: torch.Tensor,
    prefill_token_time: float,
    request_start_time: float,
):
    """Run the entire decode loop in a plain OS thread (no asyncio).

    After prefill completes on the event loop, this function takes over.
    Each step is: generate inputs -> shm decode_step (futex
    write+wait) -> record metrics.  No event-loop round-trips means
    zero queuing delay between concurrent requests.
    """
    last_token_time = prefill_token_time
    token_idx = 1
    acoustic_tokens = prefill_acoustic_tokens

    try:
        for _ in range(output_steps - 1):
            step_acoustic_tokens = acoustic_tokens[-1:]
            current_text_token = torch.randint(
                0, 10000, (1,), dtype=torch.int32
            )
            custom_outputs = engine.decode_step_shm(
                request_id,
                custom_inputs={
                    "acoustic_tokens": step_acoustic_tokens,
                    "text_tokens": current_text_token,
                    "text_mask": torch.ones(1, dtype=model_dtype),
                    "bos_mask": torch.zeros(1, dtype=model_dtype),
                },
            )
            acoustic_tokens = custom_outputs.get("acoustic_tokens")
            now = time.perf_counter()

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
    metrics: Dict[str, Any],
    request_id: str,
):
    """
    SHM mode: prefill via ZMQ (add_request on event loop), then the
    entire decode loop runs in a dedicated OS thread — no event-loop
    round-trips between steps.
    """
    model_dtype = engine.vllm_config.model_config.dtype
    prompt_acoustic_tokens = torch.randint(
        0, 1024, (input_num_tokens, 31), dtype=torch.int32
    )
    bos_mask = torch.zeros(input_num_tokens, dtype=model_dtype)
    bos_mask[0] = 1.0

    inputs = {
        "prompt_token_ids": [0] * input_num_tokens,
        "custom_inputs": {
            "acoustic_tokens": prompt_acoustic_tokens,
            "text_tokens": torch.zeros(input_num_tokens, dtype=torch.int32),
            "text_mask": torch.zeros(input_num_tokens, dtype=model_dtype),
            "bos_mask": bos_mask,
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

        acoustic_tokens = prefill_output.outputs[0].custom_outputs.get(
            "acoustic_tokens"
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _shm_thread_pool,
            _decode_loop_sync,
            engine,
            output_steps,
            model_dtype,
            metrics,
            request_id,
            acoustic_tokens,
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

    # First 6 tokens (0-5) - mean time for each position
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


async def worker(
    worker_id: int,
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    output_steps: int,
    metrics: Dict[str, Any],
    use_shm: bool = False,
):
    while True:
        async with metrics["lock"]:
            if metrics["requests_to_run"] <= 0:
                break
            metrics["requests_to_run"] -= 1

        request_id = f"benchmark-w{worker_id}-{uuid.uuid4()}"
        if use_shm:
            await run_request_shm(
                engine, sampling_params, input_num_tokens,
                output_steps, metrics, request_id
            )
        else:
            await run_request(
                engine, sampling_params, input_num_tokens,
                output_steps, metrics, request_id
            )


def init_metrics(num_requests: int):
    return {
        "request_latencies": [],
        "inter_token_latencies": [],  # For tokens 6+
        "first_tokens": [[] for _ in range(6)],  # Separate list for tokens 0-5
        "completed_sequences": 0,
        "failed_sequences": 0,
        "total_tokens": 0,
        "requests_to_run": num_requests,
        "lock": asyncio.Lock(),
    }


async def main():
    parser = argparse.ArgumentParser(description="vLLM EarTTS Benchmarking Script")
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=16,
        help="Number of concurrent workers (N)",
    )
    parser.add_argument(
        "-m",
        "--num-requests",
        type=int,
        default=512,
        help="Total number of requests to send (M)",
    )
    parser.add_argument(
        "-g", "--guidance-scale", type=float, default=None, help="Guidance scale"
    )
    parser.add_argument(
        "-t", "--dtype", type=str, default="float32", help="dtype of the engine",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "-i", "--input-len", type=int, default=128, help="Constant size prompt length"
    )
    parser.add_argument(
        "-o", "--output-len", type=int, default=256, help="Output tokens at 12.5Hz rate"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model length (context size)",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.7, help="GPU memory utilization (0.0 to 1.0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="eartts_vllm_model",
        help="Path to the EarTTS model directory",
    )
    parser.add_argument(
        "--input-coalesce-timeout-ms",
        type=float,
        default=0,
        help="Wait up to this many ms for all requests to receive custom "
             "inputs before running a forward pass (0 to disable)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with torch profiler",
    )
    parser.add_argument(
        "--use-shm",
        action="store_true",
        help="Use shared-memory decode channel (prefill via ZMQ, "
             "decode via SHM)",
    )
    args = parser.parse_args()

    global _shm_thread_pool
    _shm_thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency,
        thread_name_prefix="shm-decode",
    )

    mode = "SHM" if args.use_shm else "ZMQ"
    print(
        f"Benchmark ({mode}): concurrency={args.concurrency}, "
        f"requests={args.num_requests}, "
        f"in={args.input_len}, out={args.output_len}"
        + (" [profiling enabled]" if args.profile else "")
    )

    engine_args_kwargs: Dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_mem,
        "skip_tokenizer_init": True,
        "load_format": "dummy",
        "disable_log_stats": True,
        "enable_prefix_caching": False,
        "shm_decode": args.use_shm,
        "input_coalesce_timeout_ms": args.input_coalesce_timeout_ms,
    }

    engine_args = AsyncEngineArgs(**engine_args_kwargs)

    print("Initializing engine...")
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_args: Dict[str, Any] = {
        "max_tokens": args.max_model_len,
        "skip_sampling": True,
    }
    if args.guidance_scale is not None:
        sampling_args["guidance_scale"] = args.guidance_scale
    sampling_params = SamplingParams(**sampling_args)

    for run, num_requests in enumerate([3 * args.concurrency, args.num_requests]):
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
                        metrics=metrics,
                        use_shm=args.use_shm,
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
        print("Benchmark interrupted.")
