#!/usr/bin/env python3
"""
Build prefill embeddings from pre-extracted reference data + target text.

This script avoids importing top-level `qwen_tts` so it can run in
environments without `torchaudio` (for example Triton inference images).
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor


# ============================================================================
# Helper MLP (same as in prefill_encoder.py — duplicated to be self-contained)
# ============================================================================

class ResizeMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int,
                 act: str = "silu", bias: bool = True):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


# ============================================================================
# PrefillAssembler — embedding-only module, no audio encoders
# ============================================================================

class PrefillAssembler(nn.Module):
    """
    Assembles the dense prefill embedding from pre-computed artefacts.

    All inputs are integer IDs or pre-computed float tensors — no raw audio.
    The module contains only embedding tables and a projection MLP, so it can
    be exported to TorchScript for lightweight per-request execution.

    Inputs (all on the same device / dtype as the module):
        speaker_embedding  : [1, D]           speaker embedding vector
        ref_audio_codes    : [T_audio, G]     discrete codes from speech tokenizer
        text_ids           : [1, T_text]      target text token IDs
        ref_text_ids       : [1, T_ref]       reference transcript token IDs
        language_id        : [1] or None      codec language id; None means auto

    Output:
        prefill_embeds     : [1, S, D]        dense embedding for the Talker
    """

    def __init__(
        self,
        text_vocab_size: int,
        text_hidden_size: int,
        codec_vocab_size: int,
        hidden_size: int,
        num_code_groups: int,
        sub_codec_vocab_size: int,
        codec_pad_id: int,
        codec_bos_id: int,
        codec_think_id: int,
        codec_nothink_id: int,
        codec_think_bos_id: int,
        codec_think_eos_id: int,
        tts_bos_token_id: int,
        tts_eos_token_id: int,
        tts_pad_token_id: int,
        hidden_act: str = "silu",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_code_groups = num_code_groups
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.codec_think_id = codec_think_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.tts_pad_token_id = tts_pad_token_id

        self.text_embedding = nn.Embedding(text_vocab_size, text_hidden_size)
        self.text_projection = ResizeMLP(
            text_hidden_size, text_hidden_size, hidden_size, hidden_act, bias=True,
        )
        self.codec_embedding = nn.Embedding(codec_vocab_size, hidden_size)
        self.sub_codec_embeddings = nn.ModuleList([
            nn.Embedding(sub_codec_vocab_size, hidden_size)
            for _ in range(num_code_groups - 1)
        ])
        self.register_buffer("role_prefix_ids", torch.zeros((1, 3), dtype=torch.long), persistent=True)

    def set_role_prefix_ids(self, role_prefix_ids: torch.Tensor) -> None:
        role_prefix_ids = role_prefix_ids.to(device=self.role_prefix_ids.device, dtype=torch.long)
        if role_prefix_ids.dim() == 1:
            role_prefix_ids = role_prefix_ids.unsqueeze(0)
        if role_prefix_ids.shape != (1, 3):
            raise ValueError(f"Expected role_prefix_ids shape [1, 3], got {tuple(role_prefix_ids.shape)}")
        self.role_prefix_ids.copy_(role_prefix_ids)

    def _embed_ref_codes(self, ref_audio_codes: torch.Tensor) -> torch.Tensor:
        """Sum codebook embeddings per timestep.  ``[T, G] → [1, T, D]``."""
        parts = [self.codec_embedding(ref_audio_codes[:, 0:1])]
        for i, embed in enumerate(self.sub_codec_embeddings):
            parts.append(embed(ref_audio_codes[:, i + 1 : i + 2]))
        return torch.cat(parts, dim=1).sum(dim=1).unsqueeze(0)

    def forward(
        self,
        speaker_embedding: torch.Tensor,
        ref_audio_codes: torch.Tensor,
        text_ids: torch.Tensor,
        ref_text_ids: torch.Tensor,
        language_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = self.codec_embedding.weight.device

        # --- TTS special-token embeddings ---
        special_ids = torch.tensor(
            [[self.tts_bos_token_id, self.tts_eos_token_id,
              self.tts_pad_token_id]],
            device=device, dtype=torch.long,
        )
        special = self.text_projection(self.text_embedding(special_ids))
        tts_bos = special[:, 0:1]
        tts_eos = special[:, 1:2]
        tts_pad = special[:, 2:3]

        # A. Role prefix (pure text, no codec side)
        role = self.text_projection(
            self.text_embedding(self.role_prefix_ids.to(device=device)))  # [1, 3, D]

        # B+C. Codec control:
        # - language_id is None: AUTO/NOTHINK mode [nothink_id, think_bos, think_eos]
        # - language_id provided: THINK mode [think_id, think_bos, language_id, think_eos]
        if language_id is None:
            codec_header_ids = torch.tensor(
                [[self.codec_nothink_id, self.codec_think_bos_id, self.codec_think_eos_id]],
                device=device,
                dtype=torch.long,
            )  # [1, 3]
        else:
            lang = language_id.view(1, 1).to(device=device, dtype=torch.long)
            if bool((lang < 0).item()):
                codec_header_ids = torch.tensor(
                    [[self.codec_nothink_id, self.codec_think_bos_id, self.codec_think_eos_id]],
                    device=device,
                    dtype=torch.long,
                )  # [1, 3]
            else:
                header_ids = torch.tensor(
                    [[self.codec_think_id, self.codec_think_bos_id, self.codec_think_eos_id]],
                    device=device,
                    dtype=torch.long,
                )
                codec_header_ids = torch.cat(
                    [header_ids[:, :2], lang, header_ids[:, 2:]], dim=1
                )  # [1, 4]
        header_embed = self.codec_embedding(codec_header_ids)  # [1, H, D]
        suffix_ids = torch.tensor(
            [[self.codec_pad_id, self.codec_bos_id]],
            device=device, dtype=torch.long,
        )
        suffix_embed = self.codec_embedding(suffix_ids)        # [1, 2, D]
        spk = speaker_embedding.view(1, 1, -1)                # [1, 1, D]
        codec_ctrl = torch.cat(
            [header_embed, spk, suffix_embed], dim=1)          # [1, N, D]

        n_ctrl = codec_ctrl.shape[1]
        text_ctrl = torch.cat([
            tts_pad.expand(-1, n_ctrl - 2, -1),
            tts_bos,
        ], dim=1)                                              # [1, N-1, D]
        ctrl = text_ctrl + codec_ctrl[:, :-1]                  # [1, N-1, D]

        # D. Text: ref_text + synth_text + TTS-EOS, paired with codec_pad.
        combined_text = torch.cat([ref_text_ids, text_ids], dim=-1)   # [1, T1-1]
        text_embed = self.text_projection(self.text_embedding(combined_text))
        text_embed = torch.cat([text_embed, tts_eos], dim=1)  # [1, T1, D]
        t1 = text_embed.shape[1]

        codec_pad_ids = torch.full(
            (1, t1), self.codec_pad_id, device=device, dtype=torch.long,
        )
        text_part = text_embed + self.codec_embedding(codec_pad_ids)

        # E. Ref audio codes: BOS + Σ-codebook embeddings, paired with tts_pad
        ref_embed = self._embed_ref_codes(ref_audio_codes)     # [1, T_audio, D]
        bos_id = torch.tensor(
            [[self.codec_bos_id]], device=device, dtype=torch.long,
        )
        codec_part_raw = torch.cat(
            [self.codec_embedding(bos_id), ref_embed], dim=1)  # [1, T2, D]
        t2 = codec_part_raw.shape[1]
        codec_part = codec_part_raw + tts_pad.expand(-1, t2, -1)

        return torch.cat([role, ctrl, text_part, codec_part], dim=1)

    # ------------------------------------------------------------------
    # Loading from the full TTS model
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_model(cls, model) -> "PrefillAssembler":
        """Create from a loaded Qwen3TTSForConditionalGeneration model."""
        cfg = model.config
        tc = cfg.talker_config
        talker = model.talker

        assembler = cls(
            text_vocab_size=talker.model.text_embedding.num_embeddings,
            text_hidden_size=tc.text_hidden_size,
            codec_vocab_size=tc.vocab_size,
            hidden_size=tc.hidden_size,
            num_code_groups=tc.num_code_groups,
            sub_codec_vocab_size=talker.code_predictor.model.codec_embedding[0].num_embeddings,
            codec_pad_id=tc.codec_pad_id,
            codec_bos_id=tc.codec_bos_id,
            codec_think_id=tc.codec_think_id,
            codec_nothink_id=tc.codec_nothink_id,
            codec_think_bos_id=tc.codec_think_bos_id,
            codec_think_eos_id=tc.codec_think_eos_id,
            tts_bos_token_id=cfg.tts_bos_token_id,
            tts_eos_token_id=cfg.tts_eos_token_id,
            tts_pad_token_id=cfg.tts_pad_token_id,
            hidden_act=tc.hidden_act,
        )

        assembler.text_embedding.load_state_dict(
            talker.model.text_embedding.state_dict())
        assembler.text_projection.load_state_dict(
            talker.text_projection.state_dict())
        assembler.codec_embedding.load_state_dict(
            talker.model.codec_embedding.state_dict())
        for i in range(assembler.num_code_groups - 1):
            assembler.sub_codec_embeddings[i].load_state_dict(
                talker.code_predictor.model.codec_embedding[i].state_dict())

        return assembler


def load_model_and_processor(
    model_path: str,
    device: str,
    dtype: torch.dtype,
):
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
    )
    processor = Qwen3TTSProcessor.from_pretrained(model_path)
    return model, processor


# ============================================================================
# Text tokenization (runs outside the model — not traceable)
# ============================================================================

def tokenize_target_text(
    processor,
    synth_text: str,
) -> torch.Tensor:
    """
    Tokenize target synthesis text into token IDs.

    This runs *before* the PrefillAssembler and is intentionally kept as a
    plain function (not part of the nn.Module) since HF tokenizers are not
    TorchScript-exportable.

    Args:
        processor: Qwen3TTS processor / tokenizer.
        synth_text: The text to synthesize.

    Returns:
        text_ids: ``[1, T_text]`` int64 tensor (CPU).
    """
    synth_full = f"<|im_start|>assistant\n{synth_text}<|im_end|>\n<|im_start|>assistant\n"
    tok = processor(text=synth_full, return_tensors="pt", padding=True)
    full_ids = tok["input_ids"]
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    text_ids = full_ids[:, 3:-5].to(torch.long)
    return text_ids


def get_codec_language_mapping(model) -> Dict[str, int]:
    mapping = getattr(model.config.talker_config, "codec_language_id", None)
    if not isinstance(mapping, dict) or len(mapping) == 0:
        raise ValueError(
            "Model config does not contain a valid talker_config.codec_language_id mapping."
        )
    return {str(name).lower(): int(lang_id) for name, lang_id in mapping.items()}


def print_codec_language_mapping(mapping: Dict[str, int]) -> None:
    print("Available codec language mappings (language -> id):")
    print("  auto: -1")
    for language, lang_id in sorted(mapping.items(), key=lambda item: (item[1], item[0])):
        print(f"  {language}: {lang_id}")


def resolve_codec_language_id(language: str, mapping: Dict[str, int]) -> int:
    key = language.strip().lower()
    if key == "auto":
        return None
    if key in mapping:
        return mapping[key]
    available = ", ".join(["auto"] + sorted(mapping.keys()))
    raise ValueError(f"Unknown language '{language}'. Supported languages: {available}")


# ============================================================================
# TorchScript export
# ============================================================================

def export_torchscript(
    assembler: PrefillAssembler,
    torchscript_path: str,
    freeze: bool = True,
) -> str:
    """
    Export the PrefillAssembler to a TorchScript module with weights.

    Returns the path to the exported file.
    """
    ts_file = Path(torchscript_path)
    ts_file.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        scripted = torch.jit.script(assembler.eval())
        if freeze:
            scripted = torch.jit.freeze(scripted.eval())
        torch.jit.save(scripted, str(ts_file))

    print(f"Exported TorchScript model to {ts_file}")
    return str(ts_file)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prefill embeddings and optionally export a TorchScript PrefillAssembler."
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--ref-data", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--language",
        default="auto",
        help="Codec language string (e.g. 'english') or 'auto'.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--torchscript-path",
        default=None,
        help="Optional output path for scripted PrefillAssembler (.pt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_path} (torchaudio-free import path) ...")
    model, processor = load_model_and_processor(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )

    assembler = PrefillAssembler.from_pretrained_model(model).to(args.device).to(dtype).eval()
    print(
        f"PrefillAssembler: hidden_size={assembler.hidden_size} "
        f"num_code_groups={assembler.num_code_groups}"
    )
    language_mapping = get_codec_language_mapping(model)
    print_codec_language_mapping(language_mapping)
    language_id = resolve_codec_language_id(args.language, language_mapping)
    print(f"Resolved language '{args.language}' -> language_id={language_id}")

    ref_data = torch.load(args.ref_data, map_location="cpu", weights_only=False)
    print(f"Loaded reference data from {args.ref_data}")
    assembler.set_role_prefix_ids(ref_data["role_prefix_ids"])

    text_ids = tokenize_target_text(processor=processor, synth_text=args.text)
    ref_text_ids = ref_data["ref_text_ids"].to(torch.long)

    if language_id:
        language_id = torch.tensor([language_id], device=args.device, dtype=torch.long)

    inputs = {
        "speaker_embedding": ref_data["speaker_embedding"].unsqueeze(0).to(args.device, dtype),
        "ref_audio_codes": ref_data["ref_audio_codes"].to(args.device, torch.long),
        "text_ids": text_ids.to(args.device, torch.long),
        "ref_text_ids": ref_text_ids.to(args.device, torch.long),
    }

    with torch.inference_mode():
        prefill = assembler(
            inputs["speaker_embedding"],
            inputs["ref_audio_codes"],
            inputs["text_ids"],
            inputs["ref_text_ids"],
            language_id,
        )

    print(f"Prefill embeds: {prefill.shape} (target tokens={inputs['text_ids'].shape[1]})")

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prefill_embeds": prefill.cpu(),
            "text": args.text,
            "language": args.language,
            "language_id": language_id,
            "ref_data_path": args.ref_data,
            "text_ids": inputs["text_ids"].cpu(),
            "ref_text_ids": inputs["ref_text_ids"].cpu(),
        },
        out_file,
    )
    print(f"Saved prefill embedding to {out_file}")

    if args.torchscript_path:
        export_torchscript(
            assembler=assembler.eval(),
            torchscript_path=args.torchscript_path,
            freeze=True,
        )


if __name__ == "__main__":
    main()
