"""
Qwen3-TTS Prefill Encoder — non-streaming, ICL voice-clone mode.

Standalone nn.Module that builds the dense embedding tensor fed as
`inputs_embeds` into the Talker backbone's first forward pass (KV-cache seed).

Includes:
  - Speech tokenizer encoder: audio waveform → discrete codes
  - Speaker encoder (ECAPA-TDNN): audio waveform → speaker embedding
  - Embedding construction: text tokens + codes + speaker embedding → prefill

All sub-modules are proper nn.Module children, so the entire prefill encoder
can be traced / exported as a single graph.

================================================================================
PREFILL SEQUENCE LAYOUT  (non-streaming ICL)
================================================================================

  Every position is TEXT_EMBED + CODEC_EMBED (element-wise sum of two streams).

  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Segment            │ Len │ Text side              │ Codec side           │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ A. Role prefix     │  3  │ proj(text_emb(role))   │ —                    │
  │ B. Think/ctrl+spk  │ N-2 │ tts_pad (repeated)     │ codec_emb(think_ids) │
  │                    │     │                        │ + speaker_embed      │
  │ C. TTS BOS         │  1  │ tts_bos                │ codec_emb(pad_id)    │
  │ D. All text        │ T1  │ proj(text_emb(ref+syn))│ codec_emb(pad_id)    │
  │    + TTS EOS       │     │ + tts_eos at the end   │ (repeated)           │
  │ E. Ref audio codes │ T2  │ tts_pad (repeated)     │ bos + Σ codebooks    │
  └──────────────────────────────────────────────────────────────────────────┘

  Total S = 3 + (N-2) + 1 + T1 + T2

================================================================================
USAGE
================================================================================

  tts = Qwen3TTSModel.from_pretrained(...)
  encoder = Qwen3TTSPrefillEncoder.from_pretrained_model(tts.model)
  encoder = encoder.to(device).to(dtype).eval()

  inputs = prepare_inputs(
      processor=tts.processor,
      speech_tokenizer=tts.model.speech_tokenizer,
      synth_text="Hello!",
      ref_text="Reference transcript",
      ref_audio_path="ref.wav",
      language="auto",
      talker_config=tts.model.config.talker_config,
      speaker_encoder_sample_rate=tts.model.speaker_encoder_sample_rate,
      device=device,
  )

  with torch.no_grad():
      prefill_embeds = encoder(**inputs)  # [1, S, D]

  talker.generate(inputs_embeds=prefill_embeds, ...)
"""

import os
import sys
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig


# ============================================================================
# Input preparation (non-traceable — tokenizer + audio I/O)
# ============================================================================

def prepare_inputs(
    processor,
    speech_tokenizer,
    synth_text: str,
    ref_text: str,
    ref_audio_path: str,
    language: str,
    talker_config,
    speaker_encoder_sample_rate: int = 24000,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Tokenize texts and preprocess audio for both speech tokenizer and speaker
    encoder.  Returns a dict whose keys match ``forward()``'s signature so you
    can call ``encoder(**prepare_inputs(...))``.

    Args:
        processor:      Qwen3TTS text tokenizer / processor.
        speech_tokenizer: ``model.speech_tokenizer`` (Qwen3TTSTokenizer wrapper).
        synth_text:     Text to synthesize.
        ref_text:       Transcript of the reference audio.
        ref_audio_path: Path to the reference wav file.
        language:       Language name ("english", "chinese", …) or "auto".
        talker_config:  ``model.config.talker_config``.
        speaker_encoder_sample_rate: Expected SR for the speaker encoder (24 kHz).
        device:         Target device string.
    """
    import librosa
    import numpy as np

    # --- Text tokenization ---
    synth_full = f"<|im_start|>assistant\n{synth_text}<|im_end|>\n<|im_start|>assistant\n"
    tok = processor(text=synth_full, return_tensors="pt", padding=True)
    full_ids = tok["input_ids"]
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    role_prefix_ids = full_ids[:, :3]
    text_ids = full_ids[:, 3:-5]

    ref_full = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
    ref_tok = processor(text=ref_full, return_tensors="pt", padding=True)
    ref_full_ids = ref_tok["input_ids"]
    if ref_full_ids.dim() == 1:
        ref_full_ids = ref_full_ids.unsqueeze(0)
    ref_text_ids = ref_full_ids[:, 3:-2]

    # Codec header (think-mode control tokens)
    if language.lower() == "auto":
        header = [
            talker_config.codec_nothink_id,
            talker_config.codec_think_bos_id,
            talker_config.codec_think_eos_id,
        ]
    else:
        lang_id = talker_config.codec_language_id[language.lower()]
        header = [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            lang_id,
            talker_config.codec_think_eos_id,
        ]
    codec_header_ids = torch.tensor([header], dtype=torch.long)

    # --- Audio loading ---
    wav_raw, sr_raw = librosa.load(ref_audio_path, sr=None, mono=True)

    # Speaker encoder input (24 kHz)
    if sr_raw != speaker_encoder_sample_rate:
        wav_24k = librosa.resample(
            wav_raw, orig_sr=sr_raw, target_sr=speaker_encoder_sample_rate,
        )
    else:
        wav_24k = wav_raw
    speaker_audio = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0)

    # Speech tokenizer input (at its native sample rate, via feature extractor)
    tok_sr = int(speech_tokenizer.feature_extractor.sampling_rate)
    if sr_raw != tok_sr:
        wav_tok = librosa.resample(wav_raw, orig_sr=sr_raw, target_sr=tok_sr)
    else:
        wav_tok = wav_raw
    fe_out = speech_tokenizer.feature_extractor(
        raw_audio=[wav_tok.astype(np.float32)],
        sampling_rate=tok_sr,
        return_tensors="pt",
    )
    tokenizer_audio = fe_out["input_values"].squeeze(1)        # [1, T]
    tokenizer_padding_mask = fe_out["padding_mask"].squeeze(1)  # [1, T]

    return {
        "speaker_audio": speaker_audio.to(device),
        "tokenizer_audio": tokenizer_audio.to(device),
        "tokenizer_padding_mask": tokenizer_padding_mask.to(device),
        "text_ids": text_ids.to(device),
        "ref_text_ids": ref_text_ids.to(device),
        "role_prefix_ids": role_prefix_ids.to(device),
        "codec_header_ids": codec_header_ids.to(device),
    }


# ============================================================================
# Small helper module (text → hidden)
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
# Prefill encoder
# ============================================================================

class Qwen3TTSPrefillEncoder(nn.Module):
    """
    Non-streaming prefill encoder for Qwen3-TTS ICL voice-clone mode.

    Sub-modules included:
      - ``speaker_encoder``           – ECAPA-TDNN (audio → speaker embedding)
      - ``speech_tokenizer_encoder``  – whisper-VQ or MiMi (audio → codes)
      - text / codec embedding tables + projection
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
        tts_bos_token_id: int,
        tts_eos_token_id: int,
        tts_pad_token_id: int,
        hidden_act: str = "silu",
        speaker_encoder_config: Optional[Qwen3TTSSpeakerEncoderConfig] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_code_groups = num_code_groups
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.tts_pad_token_id = tts_pad_token_id

        # -- Text / codec embedding tables --
        self.text_embedding = nn.Embedding(text_vocab_size, text_hidden_size)
        self.text_projection = ResizeMLP(
            text_hidden_size, text_hidden_size, hidden_size, hidden_act, bias=True,
        )
        self.codec_embedding = nn.Embedding(codec_vocab_size, hidden_size)
        self.sub_codec_embeddings = nn.ModuleList([
            nn.Embedding(sub_codec_vocab_size, hidden_size)
            for _ in range(num_code_groups - 1)
        ])

        # -- Speaker encoder (ECAPA-TDNN) --
        if speaker_encoder_config is not None:
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_encoder_config)
        else:
            self.speaker_encoder = None

        # Pre-computed mel filterbank for the speaker encoder (24 kHz / 128 mels)
        from librosa.filters import mel as librosa_mel_fn
        mel_fb = librosa_mel_fn(
            sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000,
        )
        self.register_buffer("_spk_mel_basis", torch.from_numpy(mel_fb).float())
        self.register_buffer("_spk_hann_window", torch.hann_window(1024))

        # -- Speech tokenizer encoder (set by from_pretrained_model) --
        self.speech_tokenizer_encoder: Optional[nn.Module] = None
        self.tokenizer_type: Optional[str] = None
        self.encoder_valid_num_quantizers: Optional[int] = None
        self.encode_downsample_rate: Optional[int] = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_model(cls, model) -> "Qwen3TTSPrefillEncoder":
        """
        Create from a loaded ``Qwen3TTSForConditionalGeneration`` whose
        ``speech_tokenizer`` has already been attached.

        The speech tokenizer encoder is stored **by reference** (not deep-copied)
        to save memory.  If you need an independent copy, deep-copy afterwards.
        """
        cfg = model.config
        tc = cfg.talker_config
        talker = model.talker

        encoder = cls(
            text_vocab_size=talker.model.text_embedding.num_embeddings,
            text_hidden_size=tc.text_hidden_size,
            codec_vocab_size=tc.vocab_size,
            hidden_size=tc.hidden_size,
            num_code_groups=tc.num_code_groups,
            sub_codec_vocab_size=talker.code_predictor.model.codec_embedding[0].num_embeddings,
            codec_pad_id=tc.codec_pad_id,
            codec_bos_id=tc.codec_bos_id,
            tts_bos_token_id=cfg.tts_bos_token_id,
            tts_eos_token_id=cfg.tts_eos_token_id,
            tts_pad_token_id=cfg.tts_pad_token_id,
            hidden_act=tc.hidden_act,
            speaker_encoder_config=cfg.speaker_encoder_config,
        )

        # Copy text / codec embedding weights
        encoder.text_embedding.load_state_dict(
            talker.model.text_embedding.state_dict())
        encoder.text_projection.load_state_dict(
            talker.text_projection.state_dict())
        encoder.codec_embedding.load_state_dict(
            talker.model.codec_embedding.state_dict())
        for i in range(encoder.num_code_groups - 1):
            encoder.sub_codec_embeddings[i].load_state_dict(
                talker.code_predictor.model.codec_embedding[i].state_dict())

        # Copy speaker encoder weights
        if model.speaker_encoder is not None:
            encoder.speaker_encoder.load_state_dict(
                model.speaker_encoder.state_dict())

        # Attach speech tokenizer encoder (by reference)
        speech_tok = model.speech_tokenizer
        assert speech_tok is not None, "model.speech_tokenizer must be loaded"
        speech_tok_model = speech_tok.model

        encoder.speech_tokenizer_encoder = speech_tok_model.encoder
        encoder.tokenizer_type = speech_tok.get_model_type()

        if encoder.tokenizer_type == "qwen3_tts_tokenizer_12hz":
            encoder.encoder_valid_num_quantizers = (
                speech_tok_model.encoder_valid_num_quantizers
            )
            encoder.encode_downsample_rate = speech_tok_model.encode_downsample_rate

        return encoder

    # ------------------------------------------------------------------
    # Speaker embedding: mel spectrogram → ECAPA-TDNN
    # ------------------------------------------------------------------
    def _compute_speaker_mel(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute 128-bin log-mel spectrogram for the speaker encoder.

        Uses registered buffers (mel filterbank + Hann window) so the
        computation is fully traceable.

        Args:
            y: ``[B, T]`` waveform at 24 kHz (any dtype — cast to buffer dtype).

        Returns:
            ``[B, T_mel, 128]`` log-mel spectrogram in buffer dtype.
        """
        n_fft, hop, win = 1024, 256, 1024
        padding = (n_fft - hop) // 2
        y = y.to(self._spk_hann_window.dtype)
        y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        spec = torch.stft(
            y, n_fft, hop_length=hop, win_length=win,
            window=self._spk_hann_window, center=False,
            pad_mode="reflect", normalized=False, onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel_spec = torch.matmul(self._spk_mel_basis, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec.transpose(1, 2)

    def _extract_speaker_embedding(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """``[1, T]`` waveform at 24 kHz → speaker embedding ``[D]``."""
        mel = self._compute_speaker_mel(audio_24k)
        return self.speaker_encoder(mel)[0]

    # ------------------------------------------------------------------
    # Speech tokenizer: audio → discrete codes
    # ------------------------------------------------------------------
    def _encode_audio(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode pre-processed audio into discrete codes.

        Args:
            input_values:  ``[1, T]`` audio at the tokenizer's native SR.
            padding_mask:  ``[1, T]`` (1 = valid, 0 = pad).

        Returns:
            ``[T_codes, num_code_groups]`` integer code tensor.
        """
        if self.tokenizer_type == "qwen3_tts_tokenizer_25hz":
            wavs = [
                input_values[i, :int(padding_mask[i].sum().item())]
                for i in range(input_values.shape[0])
            ]
            codes, codes_lens = self.speech_tokenizer_encoder.quantize_speech(wavs)
            return codes[0, :codes_lens[0]]

        if self.tokenizer_type == "qwen3_tts_tokenizer_12hz":
            encoded = self.speech_tokenizer_encoder.encode(
                input_values=input_values.unsqueeze(1), return_dict=True,
            )
            audio_codes = encoded.audio_codes[
                :, :self.encoder_valid_num_quantizers
            ]
            mask_len = int(padding_mask[0].sum().item())
            t_valid = -(-mask_len // self.encode_downsample_rate)
            return audio_codes[0, :, :t_valid].transpose(0, 1)  # [T, n_q]

        raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    # ------------------------------------------------------------------
    # Codec code embedding
    # ------------------------------------------------------------------
    def _embed_ref_codes(self, ref_audio_codes: torch.Tensor) -> torch.Tensor:
        """Sum codebook embeddings per timestep.  ``[T, G] → [1, T, D]``."""
        parts = [self.codec_embedding(ref_audio_codes[:, 0:1])]
        for i in range(self.num_code_groups - 1):
            parts.append(
                self.sub_codec_embeddings[i](ref_audio_codes[:, i + 1:i + 2])
            )
        return torch.cat(parts, dim=1).sum(dim=1).unsqueeze(0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        speaker_audio: torch.Tensor,
        tokenizer_audio: torch.Tensor,
        tokenizer_padding_mask: torch.Tensor,
        text_ids: torch.Tensor,
        ref_text_ids: torch.Tensor,
        role_prefix_ids: torch.Tensor,
        codec_header_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the complete prefill embedding (non-streaming ICL mode).

        Args:
            speaker_audio:          ``[1, T_24k]``  24 kHz waveform.
            tokenizer_audio:        ``[1, T_tok]``  audio at tokenizer SR.
            tokenizer_padding_mask: ``[1, T_tok]``  1 = valid, 0 = pad.
            text_ids:               ``[1, T_text]`` synthesis text tokens.
            ref_text_ids:           ``[1, T_ref]``  reference transcript tokens.
            role_prefix_ids:        ``[1, 3]``      role prefix token IDs.
            codec_header_ids:       ``[1, H]``      think-mode control IDs.

        Returns:
            ``[1, S, D]`` prefill embedding ready for the Talker backbone.
        """
        device = self.codec_embedding.weight.device

        # --- Speaker embedding from raw audio ---
        speaker_embedding = self._extract_speaker_embedding(speaker_audio)

        # --- Discrete codes from raw audio ---
        ref_audio_codes = self._encode_audio(
            tokenizer_audio, tokenizer_padding_mask,
        )

        # --- TTS special-token embeddings ---
        special_ids = torch.tensor(
            [[self.tts_bos_token_id, self.tts_eos_token_id,
              self.tts_pad_token_id]],
            device=device, dtype=torch.long,
        )
        special = self.text_projection(self.text_embedding(special_ids))
        tts_bos = special[:, 0:1]   # [1, 1, D]
        tts_eos = special[:, 1:2]
        tts_pad = special[:, 2:3]

        # -- A. Role prefix (pure text, no codec side) --
        role = self.text_projection(
            self.text_embedding(role_prefix_ids))              # [1, 3, D]

        # -- B+C. Codec control: [header…, speaker, PAD, BOS] --
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

        # -- D. Text: ref_text + synth_text + TTS-EOS, paired with codec_pad --
        combined_text = torch.cat(
            [ref_text_ids, text_ids], dim=-1)                  # [1, T1-1]
        text_embed = self.text_projection(
            self.text_embedding(combined_text))
        text_embed = torch.cat([text_embed, tts_eos], dim=1)  # [1, T1, D]
        t1 = text_embed.shape[1]

        codec_pad_ids = torch.full(
            (1, t1), self.codec_pad_id, device=device, dtype=torch.long,
        )
        text_part = text_embed + self.codec_embedding(codec_pad_ids)

        # -- E. Ref audio codes: BOS + Σ-codebook embeddings, paired with tts_pad --
        ref_embed = self._embed_ref_codes(ref_audio_codes)     # [1, T_audio, D]
        bos_id = torch.tensor(
            [[self.codec_bos_id]], device=device, dtype=torch.long,
        )
        codec_part_raw = torch.cat(
            [self.codec_embedding(bos_id), ref_embed], dim=1)  # [1, T2, D]
        t2 = codec_part_raw.shape[1]
        codec_part = codec_part_raw + tts_pad.expand(-1, t2, -1)

        # -- Assemble --
        return torch.cat([role, ctrl, text_part, codec_part], dim=1)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract prefill embedding for the Qwen3-TTS Talker backbone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python prefill_encoder.py \\
      --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
      --text "Hello, how are you?" \\
      --ref-text "I felt like, you know, you can be both, right?" \\
      --ref-audio scarlettjohansson.wav \\
      --output prefill.pt
""",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="HuggingFace repo ID or local checkpoint directory.")
    parser.add_argument("--text", required=True,
                        help="Text to synthesize.")
    parser.add_argument("--ref-text", required=True,
                        help="Transcript of the reference audio.")
    parser.add_argument("--ref-audio", required=True,
                        help="Path to reference audio file (wav).")
    parser.add_argument("--language", default="auto",
                        help="Target language (default: auto).")
    parser.add_argument("--output", required=True,
                        help="Path to save the prefill embedding (.pt).")
    parser.add_argument("--device", default="cuda:0",
                        help="Device (default: cuda:0).")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Dtype (default: bfloat16).")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    from qwen_tts import Qwen3TTSModel

    # ---- 1. Load full model (need processor + speech tokenizer + speaker encoder) --
    print(f"Loading model from {args.model_path} ...")
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map=args.device, dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model = tts.model

    # ---- 2. Create prefill encoder (all audio+embedding components in one Module) --
    encoder = Qwen3TTSPrefillEncoder.from_pretrained_model(model)
    encoder = encoder.to(args.device).to(dtype).eval()
    print(f"Prefill encoder:  hidden_size={encoder.hidden_size}  "
          f"num_code_groups={encoder.num_code_groups}  "
          f"tokenizer_type={encoder.tokenizer_type}")

    # ---- 3. Prepare all inputs (text tokenization + audio preprocessing) ----------
    inputs = prepare_inputs(
        processor=tts.processor,
        speech_tokenizer=model.speech_tokenizer,
        synth_text=args.text,
        ref_text=args.ref_text,
        ref_audio_path=args.ref_audio,
        language=args.language,
        talker_config=model.config.talker_config,
        speaker_encoder_sample_rate=model.speaker_encoder_sample_rate,
        device=args.device,
    )
    print(f"Text tokens:      text_ids={inputs['text_ids'].shape}  "
          f"ref_text_ids={inputs['ref_text_ids'].shape}")
    print(f"Speaker audio:    {inputs['speaker_audio'].shape}")
    print(f"Tokenizer audio:  {inputs['tokenizer_audio'].shape}")

    # ---- 4. Run prefill encoder (single forward — audio in, embedding out) --------
    with torch.no_grad():
        prefill = encoder(**inputs)

    print(f"Prefill embeds:   {prefill.shape}  "
          f"({prefill.shape[1]} positions × {prefill.shape[2]}d)")

    # ---- 5. Save ------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "prefill_embeds": prefill.cpu(),
        "text": args.text,
        "ref_text": args.ref_text,
        "ref_audio": args.ref_audio,
        "language": args.language,
    }, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
