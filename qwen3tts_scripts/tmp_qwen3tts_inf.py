#!/usr/bin/env python3
"""Standalone Qwen3-TTS inference script with NVTX markers for nsys profiling.

Uses decode_step_shm for low-latency decode steps via shared memory.

vLLM runs the EngineCore/GPU worker in a forked child process, so you need
--trace-fork-before-exec=true for nsys to capture CUDA activity.

Usage:
    nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
        -t cuda,nvtx -o qwen3_tts_profile --force-overwrite=true \
        python tmp_qwen3tts_inf.py
"""

import argparse
import asyncio
import os

os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

import torch
import torch.cuda.nvtx as nvtx
from pathlib import Path

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


async def main(profile: bool, nsys: bool):
    type_str = "bfloat16"
    max_len = 256
    config_path = Path("dummy_qwen3_tts_model")

    engine_args = AsyncEngineArgs(
        model=str(config_path.absolute()),
        dtype=type_str,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
        gpu_memory_utilization=0.6,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
        trust_remote_code=True,
        compilation_config={"cudagraph_mode": "PIECEWISE"},
        input_coalesce_timeout_ms=5,
        shm_decode=True,
    )

    print("Initializing engine...")
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(max_tokens=max_len, skip_sampling=True)
    print("Engine initialized successfully")

    prefill_data = torch.load(
        "/home/vklimkov/workspace/vllm/vllm/dummy_qwen3_tts_model/prefill_input.pt"
    )
    prefill_emb = prefill_data["talker_input_embeds"][0].contiguous().cpu()
    prompt_len = prefill_emb.shape[0]
    print(f"Prefill embedding shape: {prefill_emb.shape}")

    request_id = "test_request_1"
    inputs = {
        "prompt_token_ids": [0] * prompt_len,
        "custom_inputs": {
            "combined_embeddings": prefill_emb,
        },
    }

    codec_eos_token_id = 2150

    print(f"Starting generation – prompt_len={prompt_len}, max_len={max_len}")

    if profile:
        await engine.start_profile()
    if nsys:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()

    nvtx.range_push("generation_total")

    # Prefill: submit request and wait for the first output via the queue.
    nvtx.range_push("prefill")
    queue = await engine.add_request(request_id, inputs, sampling_params)
    prefill_output = await queue.get()
    next_input = prefill_output.outputs[0].custom_outputs[
        "next_input_embeddings"
    ][-1:, :]
    next_tokens = prefill_output.outputs[0].custom_outputs["codes"][-1:]
    nvtx.range_pop()  # prefill

    generated_codecs = []
    step_count = 0

    for i in range(max_len):
        nvtx.range_push(f"step_{i}")
        generated_codecs.append(next_tokens)

        if next_tokens[0, 0].item() == codec_eos_token_id:
            print(f"EOS at step {i + 1}, stopping.")
            nvtx.range_pop()
            await engine.abort(request_id)
            break

        if i >= max_len - prompt_len:
            nvtx.range_pop()
            await engine.abort(request_id)
            break

        outputs = engine.decode_step_shm(
            request_id,
            custom_inputs={"combined_embeddings": next_input},
        )

        next_input = outputs["next_input_embeddings"][-1:, :]
        next_tokens = outputs["codes"][-1:].clone()
        step_count += 1

        print(
            f"Step {step_count}: codecs {next_tokens.shape}, "
            f"next_input {next_input.shape}"
        )
        nvtx.range_pop()  # step

    nvtx.range_pop()  # generation_total

    if profile:
        await engine.stop_profile()
    if nsys:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

    print(f"Done – {step_count} decode steps generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="store_true", help="Enable engine profiling"
    )
    parser.add_argument(
        "--nsys", action="store_true", help="Enable nsys profiling"
    )
    args = parser.parse_args()
    asyncio.run(main(args.profile, args.nsys))
