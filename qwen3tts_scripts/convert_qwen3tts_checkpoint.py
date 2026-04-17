#!/usr/bin/env python3
"""Convert a Qwen3-TTS CustomVoice checkpoint to vLLM-compatible format.

Takes an input directory containing config.json and model.safetensors,
applies vLLM-specific config adjustments, precomputes additional weight
tensors required for CUDA-graph-safe inference, renames weights to match
the vLLM model layout, and writes everything to an output directory.

The CustomVoice model has built-in known speakers (e.g. Aiden, Vivian, Ryan)
whose embeddings are stored in the codec embedding table.  Speaker IDs are
preserved in config.json under talker_config.spk_id.

Usage:
    python convert_qwen3tts_checkpoint.py INPUT_DIR OUTPUT_DIR
"""

import json
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ── Config adjustments ───────────────────────────────────────────────


def _adjust_config(config: dict) -> None:
    """Apply vLLM-specific adjustments to the config dict (in place)."""

    if "talker_config" not in config or "hidden_size" not in config["talker_config"]:
        raise ValueError(
            "Cannot determine talker hidden_size from config.json. "
            "Ensure talker_config.hidden_size is present."
        )
    hidden_size = config["talker_config"]["hidden_size"]
    num_code_groups = config["talker_config"].get("num_code_groups", 16)

    # 1. Custom input specs: text_ids + prev_hidden
    print("  Setting custom_input_specs...")
    config["custom_input_specs"] = [
        {"name": "text_ids", "dtype": "int64"},
        {"name": "prev_hidden", "dim": hidden_size, "dtype": "bfloat16"},
    ]

    # 2. Custom output specs: codes (groups 1..N-1) + hidden
    print("  Setting custom_output_specs...")
    config["custom_output_specs"] = [
        {"name": "codes", "dim": num_code_groups - 1, "dtype": "int64"},
        {"name": "hidden", "dim": hidden_size, "dtype": "bfloat16"},
    ]

    # 3. Fix rope_scaling in talker_config
    if "talker_config" in config:
        tc = config["talker_config"]
        rs = tc.get("rope_scaling")
        if rs is not None:
            if rs.get("interleaved", False) and "mrope_interleaved" not in rs:
                rs["mrope_interleaved"] = True
            # Remove keys that HF's rope validation doesn't recognize.
            # "interleaved" is the old Qwen format; vLLM uses "mrope_interleaved".
            # "type" is legacy; vLLM uses "rope_type".
            for stale_key in ("interleaved", "type"):
                if stale_key in rs:
                    print(f"  Removing stale '{stale_key}' from rope_scaling...")
                    del rs[stale_key]

    # 4. Add top-level sampling parameters (for group-0 via vLLM sampler)
    defaults = {
        "do_sample": True,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
    }
    for key, val in defaults.items():
        if key not in config:
            config[key] = val

    # 5. Fix code_predictor_config sampling parameters.
    #    The original checkpoint has HF GenerationConfig boilerplate
    #    (do_sample=false, temperature=1.0) which doesn't reflect the
    #    actual runtime defaults used by the original HF implementation
    #    (subtalker_dosample=True, subtalker_temperature=0.9, etc.).
    if "talker_config" in config:
        cp_cfg = config["talker_config"].get("code_predictor_config", {})
        cp_sampling_defaults = {
            "do_sample": True,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        }
        for key, val in cp_sampling_defaults.items():
            cp_cfg[key] = val
        print("  Set code_predictor_config sampling parameters.")


# ── Weight computation ───────────────────────────────────────────────


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


def _create_acoustic_zero_token(
    weights: dict[str, torch.Tensor],
    config: dict,
) -> int:
    """Pick a suppressed token and zero its row in ALL codec embeddings.

    Zeroes the chosen token's embedding in the group-0 codec embedding
    (``talker.model.codec_embedding.weight``) and in each code-predictor
    codec embedding (groups 1..N-1).  If the code-predictor vocab is too
    small to contain the chosen token ID, the embedding and lm_head
    weight tensors are padded with zeroed rows and
    ``code_predictor_config.vocab_size`` is updated.

    The chosen ID is stored in ``config["talker_config"]["acoustic_zero_token_id"]``.

    Returns the chosen token ID.
    """
    tc = config["talker_config"]
    vocab_size = tc["vocab_size"]
    codec_eos_token_id = tc["codec_eos_token_id"]
    suppress_start = vocab_size - 1024

    zero_token_id = suppress_start
    if zero_token_id == codec_eos_token_id:
        zero_token_id += 1

    # Group-0 codec embedding (before renaming to talker.code_predictor.*)
    key = "talker.model.codec_embedding.weight"
    if key not in weights:
        raise KeyError(f"Expected weight '{key}' not found in checkpoint")
    weights[key][zero_token_id] = 0
    print(f"  acoustic_zero_token_id: {zero_token_id} (zeroed in {key})")

    # Groups 1..N-1: zero (and pad if needed) code-predictor codec embeddings
    cp_config = tc["code_predictor_config"]
    cp_vocab = cp_config["vocab_size"]
    num_cp_groups = cp_config["num_code_groups"] - 1  # groups 1..N-1

    need_pad = zero_token_id >= cp_vocab
    if need_pad:
        new_cp_vocab = zero_token_id + 1
        pad_rows = new_cp_vocab - cp_vocab
        print(f"  Expanding code_predictor vocab {cp_vocab} -> {new_cp_vocab} "
              f"(+{pad_rows} rows) to accommodate zero_token_id={zero_token_id}")
    else:
        new_cp_vocab = cp_vocab

    for i in range(num_cp_groups):
        emb_key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        if emb_key in weights:
            if need_pad:
                w = weights[emb_key]
                pad = torch.zeros(pad_rows, w.shape[1], dtype=w.dtype)
                weights[emb_key] = torch.cat([w, pad], dim=0)
            weights[emb_key][zero_token_id] = 0

        head_key = f"talker.code_predictor.lm_head.{i}.weight"
        if head_key in weights and need_pad:
            w = weights[head_key]
            pad = torch.zeros(pad_rows, w.shape[1], dtype=w.dtype)
            weights[head_key] = torch.cat([w, pad], dim=0)

    if need_pad:
        cp_config["vocab_size"] = new_cp_vocab
    print(f"  Zeroed {num_cp_groups} code-predictor codec embeddings "
          f"at token {zero_token_id}")

    tc["acoustic_zero_token_id"] = zero_token_id
    return zero_token_id


# ── Weight renaming ──────────────────────────────────────────────────

# (old_prefix, new_prefix) – applied in order; first match wins.
_WEIGHT_RENAME_PREFIXES: list[tuple[str, str]] = [
    # codec_embedding moved from talker.model → talker.code_predictor
    ("talker.model.codec_embedding.", "talker.code_predictor.codec_embedding."),
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

    model_type = config.get("tts_model_type", "base")
    print(f"  Model type: {model_type}")

    tc = config.get("talker_config", {})
    spk_id = tc.get("spk_id", {})
    if spk_id:
        print(f"  Known speakers ({len(spk_id)}): {', '.join(spk_id.keys())}")
    spk_is_dialect = tc.get("spk_is_dialect", {})
    if spk_is_dialect:
        dialects = {k: v for k, v in spk_is_dialect.items() if v}
        if dialects:
            print(f"  Dialect speakers: {dialects}")

    _adjust_config(config)

    # ── 2. Load weights ──────────────────────────────────────────
    sf_file = in_path / "model.safetensors"

    print(f"  Loading {sf_file} ...")
    weights = load_file(str(sf_file))

    tc = config["talker_config"]
    suppress_mask = _compute_suppress_mask(
        tc["vocab_size"], tc["codec_eos_token_id"]
    )

    print(f"  suppress_mask: shape={suppress_mask.shape}, "
            f"suppressed={suppress_mask.sum().item()} tokens")

    weights["talker.suppress_mask"] = suppress_mask

    _create_acoustic_zero_token(weights, config)

    # ── 3. Write config (after weight-derived fields are added) ──
    with open(out_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote {out_path / 'config.json'}")

    # ── 4. Rename weights to match refactored vLLM model ─────────
    weights = _rename_weights(weights)

    out_sf = out_path / "model.safetensors"
    print(f"  Saving {out_sf} ...")
    save_file(weights, str(out_sf))

    # ── 5. Copy tokenizer files so AutoTokenizer works from output_dir
    tokenizer_files = [
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    for fname in tokenizer_files:
        src = in_path / fname
        if src.exists():
            shutil.copy2(str(src), str(out_path / fname))
            print(f"  Copied {fname}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS HF checkpoint to vLLM format"
    )
    parser.add_argument("input_dir", help="Input directory with config.json + model.safetensors")
    parser.add_argument("output_dir", help="Output directory for converted checkpoint")
    args = parser.parse_args()

    convert(args.input_dir, args.output_dir)
