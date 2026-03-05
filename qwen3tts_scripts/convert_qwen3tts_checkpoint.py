#!/usr/bin/env python3
"""Convert a Qwen3-TTS HuggingFace checkpoint to vLLM-compatible format.

Takes an input directory containing config.json and model.safetensors,
applies vLLM-specific config adjustments, precomputes additional weight
tensors required for CUDA-graph-safe inference, renames weights to match
the vLLM model layout, and writes everything to an output directory.

Weight renames applied (to match the refactored vLLM model):
  - talker.model.codec_embedding.* → talker.code_predictor.codec_embedding.*
  - talker.codec_head.*            → talker.code_predictor.codec_head.*

Precomputed weights added to model.safetensors:
  - talker.tts_pad_embed                    [hidden_size]  float
      = text_projection(text_embedding(tts_pad_token_id))
      Added to codec embeddings at every autoregressive step to
      maintain the dual-stream text+codec architecture.
  - talker.code_predictor.suppress_mask     [vocab_size]   bool
      True for the top 1024 token IDs (except codec_eos_token_id).
      Used to suppress reserved/invalid tokens during sampling.

Usage:
    python convert_qwen3tts_checkpoint.py INPUT_DIR OUTPUT_DIR
"""

import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


# ── Config adjustments ───────────────────────────────────────────────


def _adjust_config(config: dict) -> None:
    """Apply vLLM-specific adjustments to the config dict (in place)."""

    # 1. Add custom_input_specs for vLLM prompt-embed support
    if "custom_input_specs" not in config:
        print("  Adding custom_input_specs...")
        dim = 2048  # fallback
        if "talker_config" in config:
            tc = config["talker_config"]
            dim = tc.get("text_hidden_size", tc.get("hidden_size", dim))
        config["custom_input_specs"] = [
            {"name": "combined_embeddings", "dim": dim}
        ]

    # 2. Add custom_outputs
    if "custom_output_specs" not in config:
        print("  Adding custom_outputs...")
        codes_num = 16
        config["custom_output_specs"] = [
            {"name": "next_input_embeddings", "dim": dim},
            {"name": "codes", "dim": codes_num, "dtype": "int64"},
        ]

    # 3. Fix rope_scaling in talker_config
    if "talker_config" in config:
        tc = config["talker_config"]
        rs = tc.get("rope_scaling")
        if rs is not None and rs.get("interleaved", False):
            if "mrope_interleaved" not in rs:
                print("  Adding mrope_interleaved=True to rope_scaling...")
                rs["mrope_interleaved"] = True

    # 4. Add sampling parameters (original hard defaults)
    defaults = {
        "do_sample": True,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    for key, val in defaults.items():
        if key not in config:
            config[key] = val


# ── Weight computation ───────────────────────────────────────────────


def _compute_tts_pad_embed(
    weights: dict[str, torch.Tensor],
    tts_pad_token_id: int,
    hidden_act: str,
) -> torch.Tensor:
    """Compute tts_pad_embed = text_projection(text_embedding(tts_pad_token_id)).

    This runs the text_projection MLP (linear_fc1 -> act -> linear_fc2)
    on the text embedding of the pad token, entirely in float32 for
    numerical precision, then casts back to the embedding dtype.
    """
    text_emb_weight = weights["talker.model.text_embedding.weight"]
    fc1_w = weights["talker.text_projection.linear_fc1.weight"]
    fc1_b = weights["talker.text_projection.linear_fc1.bias"]
    fc2_w = weights["talker.text_projection.linear_fc2.weight"]
    fc2_b = weights["talker.text_projection.linear_fc2.bias"]

    # Look up the pad-token embedding
    x = text_emb_weight[tts_pad_token_id].float()

    # Forward through the ResizeMLP: linear_fc1 -> act -> linear_fc2
    x = x @ fc1_w.float().T + fc1_b.float()
    if hidden_act == "silu":
        x = F.silu(x)
    elif hidden_act == "gelu":
        x = F.gelu(x)
    else:
        raise ValueError(f"Unsupported hidden_act: {hidden_act}")
    x = x @ fc2_w.float().T + fc2_b.float()

    return x.to(text_emb_weight.dtype)


def _compute_suppress_mask(
    vocab_size: int,
    codec_eos_token_id: int,
) -> torch.Tensor:
    """Build a bool mask [vocab_size] that is True for suppressed tokens.

    The original Qwen3TTS model suppresses the top 1024 token IDs
    (reserved/invalid range) except for the codec EOS token.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    suppress_start = vocab_size - 1024
    if suppress_start > 0:
        mask[suppress_start:] = True
        if codec_eos_token_id >= suppress_start:
            mask[codec_eos_token_id] = False
    return mask


# ── Weight renaming ──────────────────────────────────────────────────

# (old_prefix, new_prefix) – applied in order; first match wins.
_WEIGHT_RENAME_PREFIXES: list[tuple[str, str]] = [
    # codec_embedding moved from talker.model → talker.code_predictor
    ("talker.model.codec_embedding.", "talker.code_predictor.codec_embedding."),
    # codec_head moved from talker → talker.code_predictor
    ("talker.codec_head.", "talker.code_predictor.codec_head."),
]


def _rename_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename weight keys to match the refactored vLLM model layout.

    Returns a new dict (same tensors, no copies).
    """
    renamed: dict[str, torch.Tensor] = {}
    num_renamed = 0
    for name, tensor in weights.items():
        new_name = name
        for old_pfx, new_pfx in _WEIGHT_RENAME_PREFIXES:
            if name.startswith(old_pfx):
                new_name = new_pfx + name[len(old_pfx):]
                num_renamed += 1
                break
        renamed[new_name] = tensor
    if num_renamed:
        print(f"  Renamed {num_renamed} weight key(s) to match vLLM layout.")
    return renamed


# ── Main conversion ──────────────────────────────────────────────────


def convert(input_dir: str, output_dir: str) -> None:
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Config ─────────────────────────────────────────────────
    print("Reading config.json ...")
    with open(in_path / "config.json") as f:
        config = json.load(f)

    _adjust_config(config)

    with open(out_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote {out_path / 'config.json'}")

    # ── 2. Load weights ──────────────────────────────────────────
    sf_file = in_path / "model.safetensors"

    # Single safetensors file – load all, add new tensors, save
    print(f"  Loading {sf_file} ...")
    weights = load_file(str(sf_file))

    tc = config["talker_config"]
    tts_pad_embed = _compute_tts_pad_embed(
        weights, config["tts_pad_token_id"], tc["hidden_act"]
    )
    suppress_mask = _compute_suppress_mask(
        tc["vocab_size"], tc["codec_eos_token_id"]
    )

    print(f"  tts_pad_embed: shape={tts_pad_embed.shape}, dtype={tts_pad_embed.dtype}")
    print(f"  suppress_mask: shape={suppress_mask.shape}, "
            f"suppressed={suppress_mask.sum().item()} tokens")

    weights["talker.tts_pad_embed"] = tts_pad_embed
    weights["talker.code_predictor.suppress_mask"] = suppress_mask

    # ── 3. Rename weights to match refactored vLLM model ─────────
    weights = _rename_weights(weights)

    out_sf = out_path / "model.safetensors"
    print(f"  Saving {out_sf} ...")
    save_file(weights, str(out_sf))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS HF checkpoint to vLLM format"
    )
    parser.add_argument("input_dir", help="Input directory with config.json + model.safetensors")
    parser.add_argument("output_dir", help="Output directory for converted checkpoint")
    args = parser.parse_args()

    convert(args.input_dir, args.output_dir)
