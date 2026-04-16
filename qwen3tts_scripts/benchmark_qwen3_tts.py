"""Benchmark script for Qwen3 TTS model on vLLM (SHM decode).

Reads texts from a file (one utterance per line, tab-separated with text in the
second column), tokenises them into real prefill inputs, and runs concurrent
SHM-decode inference through the vLLM engine.

Usage:
    python benchmarks/benchmark_qwen3_tts.py \
        --model /path/to/qwen3_tts_checkpoint \
        --text-file texts.txt \
        --num-requests 100 \
        --concurrency 8

    # With torch profiler:
    VLLM_TORCH_PROFILER_DIR=/tmp/traces python benchmarks/benchmark_qwen3_tts.py \
        --model /path/to/checkpoint --text-file texts.txt --profile
"""

import os

os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

import argparse
import asyncio
import concurrent.futures
import json
import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logging.getLogger("vllm").setLevel(logging.WARNING)

from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.model_executor.models.qwen3_tts import (
    Qwen3TTSTalkerForConditionalGeneration,
)


def _build_request_inputs(
    tokenizer,
    config: dict,
    text: str,
    speaker: str,
    language: str,
) -> dict:
    """Build vLLM engine inputs from a text string."""
    tc = config["talker_config"]
    hidden_size = tc["hidden_size"]

    text_ids, group0_ids = (
        Qwen3TTSTalkerForConditionalGeneration.build_prefill_tokens(
            tokenizer=tokenizer,
            text=text,
            speaker=speaker,
            language=language,
            config=config,
        )
    )

    prompt_len = text_ids.shape[0]

    inputs = {
        "prompt_token_ids": group0_ids.tolist(),
        "custom_inputs": {
            "text_ids": text_ids,
            "prev_hidden": torch.zeros(
                prompt_len, hidden_size, dtype=torch.bfloat16
            ),
        },
    }
    return inputs


def _decode_loop_sync(
    engine: AsyncLLM,
    config: dict,
    max_decode_steps: int,
    metrics: Dict[str, Any],
    request_id: str,
    prefill_custom_outputs: Dict[str, Any],
    prefill_token_time: float,
    request_start_time: float,
):
    """Run the entire decode loop in a plain OS thread (no asyncio)."""
    tc = config["talker_config"]
    codec_eos_token_id = tc["codec_eos_token_id"]
    tts_pad_token_id = config["tts_pad_token_id"]

    prev_hidden = prefill_custom_outputs["hidden"][-1:].clone()

    decode_text_id = torch.tensor([tts_pad_token_id], dtype=torch.long)

    last_token_time = prefill_token_time
    token_idx = 1

    try:
        for step in range(max_decode_steps - 1):
            custom_outputs = engine.decode_step_shm(
                request_id,
                custom_inputs={
                    "text_ids": decode_text_id,
                    "prev_hidden": prev_hidden,
                },
            )
            now = time.perf_counter()

            prev_hidden = custom_outputs["hidden"][-1:].clone()

            sampled_token = custom_outputs.get("sampled_token_ids")
            g0_tok = int(sampled_token[-1])

            if g0_tok == codec_eos_token_id:
                break

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


async def run_request(
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    config: dict,
    max_decode_steps: int,
    metrics: Dict[str, Any],
    request_id: str,
    inputs: dict,
):
    """Prefill via ZMQ, then decode loop in a dedicated OS thread."""
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
            _thread_pool,
            _decode_loop_sync,
            engine,
            config,
            max_decode_steps,
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


def print_metrics(
    metrics: Dict[str, Any], total_time: float, args: argparse.Namespace
):
    total_sequences = metrics["completed_sequences"]
    if total_sequences == 0:
        print("Error: No sequences completed.")
        return

    print(f"\n{'='*70}")
    print(f"Benchmark Results  (concurrency={args.concurrency}, "
          f"requests={args.num_requests})")
    print(f"{'='*70}")

    print("\n--- First 6 Tokens (mean latency in ms) ---")
    for i in range(6):
        if metrics["first_tokens"][i]:
            avg_ms = np.mean(metrics["first_tokens"][i]) * 1000
            p95_ms = np.percentile(metrics["first_tokens"][i], 95) * 1000
            print(f"  Token {i}: {avg_ms:.2f} ms (P95: {p95_ms:.2f} ms)")

    if metrics["inter_token_latencies"]:
        avg_itl_ms = np.mean(metrics["inter_token_latencies"]) * 1000
        p95_itl_ms = np.percentile(
            metrics["inter_token_latencies"], 95
        ) * 1000
        print(f"\n--- Tokens 6+ ITL ---")
        print(f"  Average: {avg_itl_ms:.2f} ms (P95: {p95_itl_ms:.2f} ms)")

    latencies = sorted(metrics["request_latencies"])
    print(f"\n--- Request Latency ---")
    print(f"  min:    {latencies[0]:.3f} s")
    print(f"  median: {latencies[len(latencies) // 2]:.3f} s")
    print(f"  mean:   {np.mean(latencies):.3f} s")
    print(f"  p90:    {latencies[int(len(latencies) * 0.9)]:.3f} s")
    print(f"  p95:    {np.percentile(latencies, 95):.3f} s")
    print(f"  max:    {latencies[-1]:.3f} s")

    total_tokens = metrics["total_tokens"]
    print(f"\n--- Throughput ---")
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Completed sequences: {total_sequences}")
    print(f"  Failed sequences: {metrics['failed_sequences']}")
    print(f"  Total decode steps: {total_tokens}")
    print(f"  Throughput: {total_tokens / total_time:.2f} steps/s")
    print(f"  Sequence throughput: {total_sequences / total_time:.2f} seq/s")
    print(f"{'='*70}\n")


_thread_pool: concurrent.futures.ThreadPoolExecutor | None = None


async def worker(
    worker_id: int,
    engine: AsyncLLM,
    sampling_params: SamplingParams,
    config: dict,
    tokenizer,
    texts: list[str],
    max_decode_steps: int,
    metrics: Dict[str, Any],
    speaker: str,
    language: str,
):
    """Persistent worker that picks texts from the queue until exhausted."""
    while True:
        async with metrics["lock"]:
            if metrics["requests_to_run"] <= 0:
                break
            metrics["requests_to_run"] -= 1

        text = random.choice(texts)
        request_id = f"bench-w{worker_id}-{uuid.uuid4()}"

        inputs = _build_request_inputs(
            tokenizer, config, text, speaker, language,
        )

        await run_request(
            engine,
            sampling_params,
            config,
            max_decode_steps,
            metrics,
            request_id,
            inputs,
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
        description="vLLM Qwen3 TTS Benchmark (SHM decode)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the Qwen3-TTS model checkpoint",
    )
    parser.add_argument(
        "--text-file", type=str, required=True,
        help="Path to text file (one utterance per line, "
             "tab-separated with text in 2nd column)",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=16,
        help="Number of concurrent workers (default: 16)",
    )
    parser.add_argument(
        "-m", "--num-requests", type=int, default=100,
        help="Total number of requests to send (default: 100)",
    )
    parser.add_argument(
        "--max-decode-steps", type=int, default=2000,
        help="Max decode steps per request before forced stop (default: 2000)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="Maximum model context length (default: 2048)",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.7,
        help="GPU memory utilization 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="Disable CUDA graph capture and run in eager mode",
    )
    parser.add_argument(
        "--input-coalesce-timeout-ms", type=float, default=0,
        help="Wait up to this many ms for custom inputs before forward pass",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip warmup run",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable torch profiler (set VLLM_TORCH_PROFILER_DIR for output)",
    )
    parser.add_argument(
        "--speaker", type=str, default="aiden",
        help="Speaker name (default: aiden)",
    )
    parser.add_argument(
        "--language", type=str, default="english",
        help="Language (default: english)",
    )

    args = parser.parse_args()
    model_path = Path(args.model)

    # ── Load texts ────────────────────────────────────────────────────────
    with open(args.text_file) as f:
        texts = [line.split("\t", 1)[1].strip() for line in f if line.strip()]

    if not texts:
        print(f"ERROR: no non-empty lines found in {args.text_file}")
        return

    print(f"Loaded {len(texts)} texts from {args.text_file}")

    # ── Load config & tokenizer ───────────────────────────────────────────
    with open(model_path / "config.json") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path.absolute()))

    # ── Thread pool ───────────────────────────────────────────────────────
    global _thread_pool
    _thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency,
        thread_name_prefix="shm-decode",
    )

    print(
        f"Benchmark (SHM): concurrency={args.concurrency}, "
        f"requests={args.num_requests}, "
        f"max_decode_steps={args.max_decode_steps}"
        + (" [profiling enabled]" if args.profile else "")
    )

    # ── Engine ────────────────────────────────────────────────────────────
    engine_args = AsyncEngineArgs(
        model=str(model_path.absolute()),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        skip_tokenizer_init=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        shm_decode=True,
        input_coalesce_timeout_ms=args.input_coalesce_timeout_ms,
        attention_backend="TRITON_ATTN",
    )

    print("Initializing engine...")
    engine = AsyncLLM.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        max_tokens=args.max_model_len,
        temperature=config.get("temperature", 0.9),
        top_k=config.get("top_k", 50),
        top_p=config.get("top_p", 1.0),
        repetition_penalty=config.get("repetition_penalty", 1.0),
    )

    # ── Warmup + Benchmark ────────────────────────────────────────────────
    warmup_num = 0 if args.no_warmup else 3 * args.concurrency
    for run, num_requests in enumerate([warmup_num, args.num_requests]):
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
                        config=config,
                        tokenizer=tokenizer,
                        texts=texts,
                        max_decode_steps=args.max_decode_steps,
                        metrics=metrics,
                        speaker=args.speaker,
                        language=args.language,
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
            print_metrics(metrics, total_time, args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
