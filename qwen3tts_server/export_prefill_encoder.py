#!/usr/bin/env python3
"""
Export TorchScript PrefillAssembler for Qwen3-TTS CustomVoice inference.

The PrefillAssembler builds dense prefill embeddings from a known speaker ID
and target text — no reference audio needed.  Speaker identity is a learned
embedding stored in the codec embedding table.

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
# Helper MLP
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
# PrefillAssembler — CustomVoice only
# ============================================================================

class PrefillAssembler(nn.Module):
    """
    Assembles dense prefill embedding for a known speaker (CustomVoice).

    Speaker identity comes from the codec embedding table — no reference
    audio, no speaker encoder, no speech tokenizer needed.

    Prefill layout:
        A. Role prefix          (3)     text_proj(text_emb(role))  |  —
        B+C. Ctrl + speaker     (N-1)   tts_pad/bos               |  codec header/spk/pad
        D. Synth text + EOS     (T+1)   text embeds               |  codec pad
        E. Final BOS            (1)     tts_pad                   |  codec bos

    Inputs:
        spk_id       : [1]          codec token ID for the speaker
        text_ids     : [1, T_text]  target text token IDs
        language_id  : [1] or None  codec language id; None means auto

    Output:
        prefill_embeds : [1, S, D]  dense embedding for the Talker
    """

    def __init__(
        self,
        text_vocab_size: int,
        text_hidden_size: int,
        codec_vocab_size: int,
        hidden_size: int,
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
        self.register_buffer("role_prefix_ids", torch.zeros((1, 3), dtype=torch.long))

    def forward(
        self,
        spk_id: torch.Tensor,
        text_ids: torch.Tensor,
        language_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = self.codec_embedding.weight.device

        # Speaker embedding: a single row from the codec embedding table
        speaker_embedding = self.codec_embedding(
            spk_id.view(1).to(device=device, dtype=torch.long)
        )  # [1, D]

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

        # A. Role prefix (baked into module at export time)
        role = self.text_projection(
            self.text_embedding(self.role_prefix_ids.to(device=device)))

        # B+C. Codec control header + speaker
        if language_id is None:
            codec_header_ids = torch.tensor(
                [[self.codec_nothink_id, self.codec_think_bos_id,
                  self.codec_think_eos_id]],
                device=device, dtype=torch.long,
            )
        else:
            lang = language_id.view(1, 1).to(device=device, dtype=torch.long)
            if bool((lang < 0).item()):
                codec_header_ids = torch.tensor(
                    [[self.codec_nothink_id, self.codec_think_bos_id,
                      self.codec_think_eos_id]],
                    device=device, dtype=torch.long,
                )
            else:
                header_ids = torch.tensor(
                    [[self.codec_think_id, self.codec_think_bos_id,
                      self.codec_think_eos_id]],
                    device=device, dtype=torch.long,
                )
                codec_header_ids = torch.cat(
                    [header_ids[:, :2], lang, header_ids[:, 2:]], dim=1
                )

        header_embed = self.codec_embedding(codec_header_ids)
        suffix_ids = torch.tensor(
            [[self.codec_pad_id, self.codec_bos_id]],
            device=device, dtype=torch.long,
        )
        suffix_embed = self.codec_embedding(suffix_ids)
        spk = speaker_embedding.view(1, 1, -1)
        codec_ctrl = torch.cat(
            [header_embed, spk, suffix_embed], dim=1)

        n_ctrl = codec_ctrl.shape[1]
        text_ctrl = torch.cat([
            tts_pad.expand(-1, n_ctrl - 2, -1),
            tts_bos,
        ], dim=1)
        ctrl = text_ctrl + codec_ctrl[:, :-1]

        # D. Synth text + TTS_EOS, paired with codec_pad
        text_embed = self.text_projection(self.text_embedding(text_ids))
        text_embed = torch.cat([text_embed, tts_eos], dim=1)
        t1 = text_embed.shape[1]
        codec_pad_ids = torch.full(
            (1, t1), self.codec_pad_id, device=device, dtype=torch.long,
        )
        text_part = text_embed + self.codec_embedding(codec_pad_ids)

        # E. Final position: tts_pad + codec_bos
        bos_embed = self.codec_embedding(
            torch.tensor([[self.codec_bos_id]], device=device, dtype=torch.long))
        final_pos = tts_pad + bos_embed

        return torch.cat([role, ctrl, text_part, final_pos], dim=1)

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

        return assembler


def load_model_and_processor(model_path: str, device: str, dtype: torch.dtype):
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path, device_map=device, dtype=dtype, attn_implementation="eager",
    )
    processor = Qwen3TTSProcessor.from_pretrained(model_path)
    return model, processor


# ============================================================================
# Helpers
# ============================================================================

def tokenize_target_text(processor, synth_text: str) -> torch.Tensor:
    """``synth_text`` → ``[1, T_text]`` int64 token IDs (CPU)."""
    synth_full = f"<|im_start|>assistant\n{synth_text}<|im_end|>\n<|im_start|>assistant\n"
    tok = processor(text=synth_full, return_tensors="pt", padding=True)
    full_ids = tok["input_ids"]
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    return full_ids[:, 3:-5].to(torch.long)


def compute_role_prefix_ids(processor) -> torch.Tensor:
    """Tokenize ``<|im_start|>assistant\\n`` → ``[1, 3]`` int64."""
    ids = processor.tokenizer.encode("<|im_start|>assistant\n")
    return torch.tensor([ids[:3]], dtype=torch.long)


def get_speaker_id(model, speaker_name: str) -> int:
    spk_id_map = getattr(model.config.talker_config, "spk_id", None)
    if not spk_id_map:
        raise ValueError("Model config has no talker_config.spk_id — not a CustomVoice model?")
    key = speaker_name.strip().lower()
    if key not in spk_id_map:
        available = ", ".join(sorted(spk_id_map.keys()))
        raise ValueError(f"Unknown speaker '{speaker_name}'. Available: {available}")
    return int(spk_id_map[key])


def get_codec_language_mapping(model) -> Dict[str, int]:
    mapping = getattr(model.config.talker_config, "codec_language_id", None)
    if not isinstance(mapping, dict) or len(mapping) == 0:
        raise ValueError("Model config has no valid talker_config.codec_language_id mapping.")
    return {str(name).lower(): int(lang_id) for name, lang_id in mapping.items()}


def resolve_codec_language_id(language: str, mapping: Dict[str, int]) -> Optional[int]:
    key = language.strip().lower()
    if key == "auto":
        return None
    if key in mapping:
        return mapping[key]
    available = ", ".join(["auto"] + sorted(mapping.keys()))
    raise ValueError(f"Unknown language '{language}'. Supported: {available}")


def resolve_dialect_language(model, speaker_name: str, language: str, language_id):
    """Override language_id if the speaker has a dialect and language is chinese/auto."""
    tc = model.config.talker_config
    spk_is_dialect = getattr(tc, "spk_is_dialect", {})
    if not spk_is_dialect:
        return language_id
    key = speaker_name.strip().lower()
    dialect = spk_is_dialect.get(key, False)
    if dialect and language.lower() in ("chinese", "auto"):
        lang_map = getattr(tc, "codec_language_id", {})
        if dialect in lang_map:
            return int(lang_map[dialect])
    return language_id


# ============================================================================
# TorchScript export
# ============================================================================

def export_torchscript(assembler: PrefillAssembler, torchscript_path: str,
                       freeze: bool = True) -> str:
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
        description="Export TorchScript PrefillAssembler for Qwen3-TTS CustomVoice.",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--text", required=True,
                        help="Test text to verify the assembler.")
    parser.add_argument("--speaker", default="Aiden",
                        help="Speaker name (default: Aiden).")
    parser.add_argument("--language", default="auto",
                        help="Codec language (e.g. 'english') or 'auto'.")
    parser.add_argument("--output", required=True,
                        help="Output path for test prefill embedding (.pt).")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--torchscript-path", default=None,
                        help="Output path for TorchScript PrefillAssembler (.pt).")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_path} ...")
    model, processor = load_model_and_processor(args.model_path, args.device, dtype)

    assembler = PrefillAssembler.from_pretrained_model(model).to(args.device).to(dtype).eval()
    print(f"PrefillAssembler: hidden_size={assembler.hidden_size}")

    # Speaker
    spk_id_val = get_speaker_id(model, args.speaker)
    print(f"Speaker '{args.speaker}' -> spk_id={spk_id_val}")

    # Language
    language_mapping = get_codec_language_mapping(model)
    language_id = resolve_codec_language_id(args.language, language_mapping)
    language_id = resolve_dialect_language(model, args.speaker, args.language, language_id)
    print(f"Language '{args.language}' -> language_id={language_id}")

    # Bake role prefix IDs into the module
    role_prefix_ids = compute_role_prefix_ids(processor)
    assembler.role_prefix_ids.copy_(role_prefix_ids.to(assembler.role_prefix_ids.device))
    print(f"Role prefix IDs: {assembler.role_prefix_ids.tolist()}")

    # Test forward
    text_ids = tokenize_target_text(processor, args.text)
    spk_id_tensor = torch.tensor([spk_id_val], device=args.device, dtype=torch.long)
    language_id_tensor = (
        torch.tensor([language_id], device=args.device, dtype=torch.long)
        if language_id is not None else None
    )

    with torch.inference_mode():
        prefill = assembler(
            spk_id_tensor, text_ids.to(args.device, torch.long),
            language_id_tensor,
        )

    print(f"Prefill embeds: {prefill.shape} (target tokens={text_ids.shape[1]})")

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prefill_embeds": prefill.cpu(),
        "text": args.text,
        "speaker": args.speaker,
        "spk_id": spk_id_val,
        "language": args.language,
        "language_id": language_id,
        "text_ids": text_ids.cpu(),
    }, out_file)
    print(f"Saved test prefill to {out_file}")

    if args.torchscript_path:
        export_torchscript(assembler.eval(), args.torchscript_path, freeze=True)


if __name__ == "__main__":
    main()
