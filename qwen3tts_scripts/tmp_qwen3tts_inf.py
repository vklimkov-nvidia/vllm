#!/usr/bin/env python3
"""Standalone Qwen3-TTS inference script with NVTX markers for nsys profiling.

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
# Force fork so nsys can follow the child process.
# (If CUDA gets initialized before the engine, vLLM auto-switches to spawn
#  and nsys loses the child.)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"

import torch
import torch.cuda.nvtx as nvtx
from pathlib import Path

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


async def main(profile: bool, nsys: bool):
    # ── engine setup ──────────────────────────────────────────────────────
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
    )

    print("Initializing engine...")
    engine = AsyncLLM.from_engine_args(engine_args)
    sampling_params = SamplingParams(max_tokens=max_len, skip_sampling=True)
    print("Engine initialized successfully")

    # ── load prefill embeddings ───────────────────────────────────────────
    prefill_data = torch.load(
        "/home/vklimkov/workspace/vllm/vllm/dummy_qwen3_tts_model/prefill_input.pt"
    )
    prefill_emb = prefill_data["talker_input_embeds"][0].contiguous().cpu()
    prompt_len = prefill_emb.shape[0]
    print(f"Prefill embedding shape: {prefill_emb.shape}")

    # ── prepare request ───────────────────────────────────────────────────
    request_id = "test_request_1"
    inputs = {
        "prompt_token_ids": [0] * prompt_len,
        "custom_inputs": {
            "combined_embeddings": prefill_emb,
        },
    }

    print(f"Starting generation – prompt_len={prompt_len}, max_len={max_len}")

    # ── generation loop ───────────────────────────────────────────────────
    codec_eos_token_id = 2150
    step_count = 0

    if profile:
        await engine.start_profile()
    if nsys:
        torch.cuda.synchronize()
        torch.cuda.profiler.start()


    nvtx.range_push("generation_total")

    async for output in engine.generate(
        inputs, sampling_params=sampling_params, request_id=request_id
    ):
        nvtx.range_push(f"step_{step_count}")

        codec_tokens = output.outputs[0].custom_outputs["codes"][-1:]

        if codec_tokens[0, 0].item() == codec_eos_token_id:
            print(f"EOS at step {step_count + 1}, stopping.")
            nvtx.range_pop()  # step
            await engine.abort(request_id)
            break

        next_input = output.outputs[0].custom_outputs["next_input_embeddings"]
        step_count += 1
        print(
            f"Step {step_count}: codecs {codec_tokens.shape}, "
            f"next_input {next_input.shape}"
        )

        nvtx.range_pop()  # step

        if step_count >= max_len - prompt_len:
            await engine.abort(request_id)
            break

        new_custom_inputs = {
            "combined_embeddings": next_input[-1:, :],
        }
        await engine.append_request(
            request_id=request_id, custom_inputs=new_custom_inputs
        )

    nvtx.range_pop()  # generation_total

    if profile:
        await engine.stop_profile()
    if nsys:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()

    print(f"Done – {step_count} decode steps generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable engine profiling")
    parser.add_argument("--nsys", action="store_true", help="Enable nsys profiling")
    args = parser.parse_args()
    asyncio.run(main(args.profile, args.nsys))

