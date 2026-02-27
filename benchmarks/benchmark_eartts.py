import os
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

import argparse
import asyncio
import time
import uuid
import logging
from typing import Dict, Any
import numpy as np
import torch

# Suppress verbose vLLM logging
logging.getLogger("vllm").setLevel(logging.WARNING)

try:
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
except ImportError:
    print("Error: Failed to import vllm.")
    print("Please install vllm: pip install vllm")
    exit(1)


async def run_request(
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    metrics: Dict[str, Any],
    request_id: str,
):
    """
    Sends a single request to the vLLM engine and records metrics.
    """
    # random input for eartts
    prompt_acoustic_tokens = torch.randint(
        0, 1024, (input_num_tokens, 31), dtype=torch.int32
    )
    model_type = engine.vllm_config.model_config.dtype
    bos_mask = torch.zeros(input_num_tokens, dtype=model_type)
    bos_mask[0] = 1.0

    # Prefill inputs: text_tokens and text_mask are masked (zeros) during prefill
    inputs = {
        "prompt_token_ids": [0] * input_num_tokens,
        "custom_inputs": {
            "acoustic_tokens": prompt_acoustic_tokens,
            "text_tokens": torch.zeros(input_num_tokens, dtype=torch.int32),
            "text_mask": torch.zeros(input_num_tokens, dtype=model_type),
            "bos_mask": bos_mask,
        },
    }

    request_start_time = time.perf_counter()
    last_token_time = None
    token_idx = 0  # 0-indexed token position

    try:
        # Start the generation stream
        results_generator = engine.generate(inputs, sampling_params, request_id)
        async for output in results_generator:
            now = time.perf_counter()

            # Get the acoustic tokens from custom_outputs
            acoustic_tokens = output.outputs[0].custom_outputs.get("acoustic_tokens")

            # Track first 6 tokens (0-5) separately
            if token_idx < 6:
                token_latency = now - last_token_time if last_token_time else now - request_start_time
                metrics["first_tokens"][token_idx].append(token_latency)
            elif last_token_time is not None:
                # Tokens 6+ go into regular ITL
                itl = now - last_token_time
                metrics["inter_token_latencies"].append(itl)

            last_token_time = now
            token_idx += 1

            # Check if the sequence has finished
            if output.finished:
                break
            else:
                # Prepare next decode step inputs
                step_acoustic_tokens = acoustic_tokens[-1:]
                current_text_token = torch.randint(0, 10000, (1,), dtype=torch.int32)
                new_custom_inputs = {
                    "acoustic_tokens": step_acoustic_tokens,
                    "text_tokens": current_text_token,
                    "text_mask": torch.ones(1, dtype=model_type),
                    "bos_mask": torch.zeros(1, dtype=model_type),
                }
                await engine.append_request(
                    request_id=request_id, custom_inputs=new_custom_inputs
                )

        # After the loop finishes (sequence is done)
        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time

        # Record final metrics for this request
        metrics["request_latencies"].append(request_latency)
        metrics["completed_sequences"] += 1
        metrics["total_tokens"] += token_idx

    except Exception as e:
        print(f"Request {request_id} failed: {e}")
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
    print(metrics["first_tokens"][0])
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
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
    input_num_tokens: int,
    metrics: Dict[str, Any],
):
    """
    A persistent worker that continuously sends requests until
    the global request counter reaches zero.
    """
    while True:

        # Atomically check and decrement the request counter
        async with metrics["lock"]:
            if metrics["requests_to_run"] <= 0:
                # All requests have been assigned, worker can exit
                break
            metrics["requests_to_run"] -= 1

        request_id = f"benchmark-w{worker_id}-{uuid.uuid4()}"
        # Run the request *outside* the lock
        await run_request(
            engine, sampling_params, input_num_tokens, metrics, request_id
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
    args = parser.parse_args()

    print(f"Benchmark: concurrency={args.concurrency}, requests={args.num_requests}, in={args.input_len}, out={args.output_len}")

    # 1. Create Engine Args
    #max_num_batched_tokens = args.max_model_len * args.concurrency * 2
    #if args.guidance_scale is not None:
    #    max_num_batched_tokens = max_num_batched_tokens
    engine_args = AsyncEngineArgs(
        model="eartts_vllm_model",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        #max_num_seqs=args.concurrency * 2,
        #max_num_batched_tokens=args.max_model_len * args.concurrency * 2,
        gpu_memory_utilization=args.gpu_mem,
        skip_tokenizer_init=True,  # Skip tokenizer since we're using embeddings directly
        load_format="dummy",  # <-- Use dummy weights as requested
        disable_log_stats=True,
        enable_prefix_caching=False,
    )

    # 2. Create Engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 3. Create Sampling Params
    sampling_args = {
        "max_tokens": args.output_len,
        "stop_token_ids": [],  # Ensure no default stop tokens interfere
        "skip_sampling": True,
    }
    if args.guidance_scale is not None:
        sampling_args["guidance_scale"] = args.guidance_scale
    sampling_params = SamplingParams(**sampling_args)

    # Shared metrics dictionary
    for run, num_requests in enumerate([3 * args.concurrency, args.num_requests]):
        metrics = init_metrics(num_requests)

        # --- Start Benchmark ---
        if run == 0:
            print("Warmup...")
        start_time = time.perf_counter()

        # Create and start C worker tasks
        tasks = []
        for i in range(args.concurrency):
            tasks.append(
                asyncio.create_task(
                    worker(
                        worker_id=i,
                        engine=engine,
                        sampling_params=sampling_params,
                        input_num_tokens=args.input_len,
                        metrics=metrics,
                    )
                )
            )

        # Wait for all worker tasks to finish
        # Workers will finish once metrics["requests_to_run"] hits 0
        await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        # --- End Benchmark ---

        total_time = end_time - start_time

        if run > 0:
            # Calculate and print final metrics
            calculate_and_print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
