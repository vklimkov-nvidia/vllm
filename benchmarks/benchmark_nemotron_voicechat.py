import argparse
import asyncio
import time
import uuid
import logging
from typing import Dict, List, Any
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
    context_text_tokens = torch.randint(
        0, 10000, (input_num_tokens,), dtype=torch.int32
    )
    input_acoustic_embeds = torch.randn(input_num_tokens, 4480, dtype=torch.bfloat16)
    context_token = context_text_tokens[-1:]  # 1
    bos_mask = torch.zeros(input_num_tokens, dtype=torch.float16)
    bos_mask[0] = 1.0

    # Prefill inputs: text_tokens and text_mask are masked (zeros) during prefill
    inputs = {
        "prompt_token_ids": [0] * input_num_tokens,
        "custom_inputs": {
            "input_acoustic_embeds": input_acoustic_embeds,
            "acoustic_tokens": prompt_acoustic_tokens,
            "context_text_tokens": context_text_tokens,
            "text_mask": torch.zeros(input_num_tokens, dtype=torch.float16),
            "bos_mask": bos_mask,
        },
    }

    request_start_time = time.perf_counter()
    last_token_time = None
    token_count = 0

    try:
        # Start the generation stream
        results_generator = engine.generate(inputs, sampling_params, request_id)
        async for output in results_generator:
            now = time.perf_counter()

            # Get the acoustic tokens from custom_outputs
            acoustic_tokens = output.outputs[0].custom_outputs.get("acoustic_tokens")
            token_count += 1

            if last_token_time is not None:
                # This is not the first token, record inter-token latency
                itl = now - last_token_time
                metrics["inter_token_latencies"].append(itl)
            else:
                # This is the first token, record time-to-first-token
                ttft = now - request_start_time
                metrics["ttft_latencies"].append(ttft)

            last_token_time = now

            # Check if the sequence has finished
            # With dummy weights, this will trigger at sampling_params.max_tokens
            if output.finished:
                break
            else:
                # Prepare next decode step inputs
                # Get the last predicted acoustic token
                step_acoustic_tokens = acoustic_tokens[-1:]
                current_text_token = torch.randint(0, 10000, (1,), dtype=torch.int32)
                new_custom_inputs = {
                    "input_acoustic_embeds": torch.randn(1, 4480, dtype=torch.bfloat16),
                    "acoustic_tokens": step_acoustic_tokens,
                    "context_text_tokens": context_token,
                    "text_mask": torch.ones(1, dtype=torch.float16),
                    "bos_mask": torch.zeros(1, dtype=torch.float16),
                }
                await engine.append_request(
                    request_id=request_id, custom_inputs=new_custom_inputs
                )
                context_token = current_text_token

        # After the loop finishes (sequence is done)
        request_end_time = time.perf_counter()
        request_latency = request_end_time - request_start_time

        # Record final metrics for this request
        metrics["request_latencies"].append(request_latency)
        metrics["completed_sequences"] += 1
        metrics["total_tokens"] += token_count

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

    total_tokens = metrics["total_tokens"]

    avg_seq_per_sec = total_sequences / total_time
    avg_token_per_sec = total_tokens / total_time
    avg_seq_len = total_tokens / total_sequences

    avg_seq_latency_s = np.mean(metrics["request_latencies"])
    p95_seq_latency_s = np.percentile(metrics["request_latencies"], 95)

    avg_ttft_ms = np.mean(metrics["ttft_latencies"]) * 1000
    p95_ttft_ms = np.percentile(metrics["ttft_latencies"], 95) * 1000

    avg_itl_ms = np.mean(metrics["inter_token_latencies"]) * 1000
    p95_itl_ms = np.percentile(metrics["inter_token_latencies"], 95) * 1000

    print("\n--- vLLM Benchmark Results ---")
    print(f"Concurrency: {args.concurrency} workers")
    print(f"Input Length: {args.input_len} tokens")
    print(f"Output Length: {args.output_len} tokens")
    print("---")
    print(f"Total duration: {total_time:.2f} s")
    print(f"Total completed sequences: {total_sequences}")
    print(f"Total failed sequences: {metrics['failed_sequences']}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average sequence length: {avg_seq_len:.2f} tokens")
    print("--- Throughput ---")
    print(f"Average sequences/sec: {avg_seq_per_sec:.2f}")
    print(f"Average tokens/sec: {avg_token_per_sec:.2f}")
    print("--- Latency ---")
    print(f"Average sequence latency: {avg_seq_latency_s:.2f} s")
    print(f"P95 sequence latency: {p95_seq_latency_s:.2f} s")
    print(f"Average TTFT (Time-To-First-Token): {avg_ttft_ms:.2f} ms")
    print(f"P95 TTFT: {p95_ttft_ms:.2f} ms")
    print(f"Average ITL (Inter-Token Latency): {avg_itl_ms:.2f} ms")
    print(f"P95 ITL: {p95_itl_ms:.2f} ms")
    print("------------------------------")


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
        "inter_token_latencies": [],
        "ttft_latencies": [],
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

    print("Starting vLLM benchmark...")
    print(f"Concurrency: {args.concurrency}, Num Requests: {args.num_requests}")
    print(
        f"Input: {args.input_len}, Output: {args.output_len}, Max Model Len: {args.max_model_len}"
    )

    # 1. Create Engine Args
    engine_args = AsyncEngineArgs(
        model="eartts_vllm_model",
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len * args.concurrency,
        gpu_memory_utilization=args.gpu_mem,
        skip_tokenizer_init=True,  # Skip tokenizer since we're using embeddings directly
        load_format="dummy",  # <-- Use dummy weights as requested
        disable_log_stats=True,
    )

    # 2. Create Engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 3. Create Sampling Params
    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        stop_token_ids=[],  # Ensure no default stop tokens interfere
        skip_sampling=True,
    )

    # Shared metrics dictionary
    for run, num_requests in enumerate([3 * args.concurrency, args.num_requests]):
        metrics = init_metrics(num_requests)

        # --- Start Benchmark ---
        print(
            f"\nStarting {run} benchmark for {num_requests} requests... This may take a moment."
        )
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

        print(f"\nBenchmark finished. Total time: {total_time:.2f}s")

        if run > 0:
            # Calculate and print final metrics
            calculate_and_print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Benchmark interrupted.")
