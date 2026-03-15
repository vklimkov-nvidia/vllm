"""
Extract reference speaker data for Qwen3-TTS prefill encoder.

Given a reference audio file and its transcript, this script:
  1. Extracts speaker embedding via ECAPA-TDNN
  2. Extracts discrete audio codes via the speech tokenizer
  3. Tokenizes the reference text and role prefix

The output .pt file contains everything needed to build prefill embeddings
for *any* target text with this speaker's voice — no model reload required.

Usage:
  python extract_reference.py \
      --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --ref-text "I felt like, you know, you can be both, right?" \
      --ref-audio scarlettjohansson.wav \
      --output ref_data.pt
"""

import os
import sys
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_encoder import Qwen3TTSPrefillEncoder


def extract_reference(
    model_path: str,
    ref_text: str,
    ref_audio_path: str,
    language: str = "auto",
    device: str = "cuda:0",
    dtype_str: str = "bfloat16",
) -> dict:
    """
    Load the full TTS model, extract all reference-speaker artefacts, and
    return them as CPU tensors in a dict suitable for ``torch.save``.

    Returns dict with keys:
        speaker_embedding  : [D]                float32
        ref_audio_codes    : [T_codes, G]       int64
        ref_text_ids       : [1, T_ref]         int64
        role_prefix_ids    : [1, 3]             int64
        codec_header_ids   : [1, H]             int64
        metadata           : dict (ref_text, ref_audio, language, model_path,
                                   hidden_size, num_code_groups, codec_pad_id,
                                   codec_bos_id, tts_bos_token_id,
                                   tts_eos_token_id, tts_pad_token_id)
    """
    import librosa
    from qwen_tts import Qwen3TTSModel

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]

    # ---- Load model --------------------------------------------------------
    print(f"Loading model from {model_path} ...")
    tts = Qwen3TTSModel.from_pretrained(
        model_path, device_map=device, dtype=dtype,
        attn_implementation="eager",
    )
    model = tts.model
    processor = tts.processor
    tc = model.config.talker_config
    spk_sr = model.speaker_encoder_sample_rate

    # ---- Build prefill encoder (for speaker enc + speech tokenizer) --------
    encoder = Qwen3TTSPrefillEncoder.from_pretrained_model(model)
    encoder = encoder.to(device).to(dtype).eval()

    # ---- Tokenize reference text ------------------------------------------
    ref_full = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
    ref_tok = processor(text=ref_full, return_tensors="pt", padding=True)
    ref_full_ids = ref_tok["input_ids"]
    if ref_full_ids.dim() == 1:
        ref_full_ids = ref_full_ids.unsqueeze(0)
    ref_text_ids = ref_full_ids[:, 3:-2]

    # Role prefix (always the same 3 tokens: <|im_start|> assistant \n)
    role_prefix_ids = ref_full_ids[:, :3]

    # Codec header (think-mode control tokens)
    if language.lower() == "auto":
        header = [
            tc.codec_nothink_id,
            tc.codec_think_bos_id,
            tc.codec_think_eos_id,
        ]
    else:
        lang_id = tc.codec_language_id[language.lower()]
        header = [
            tc.codec_think_id,
            tc.codec_think_bos_id,
            lang_id,
            tc.codec_think_eos_id,
        ]
    codec_header_ids = torch.tensor([header], dtype=torch.long)

    # ---- Load & resample audio --------------------------------------------
    wav_raw, sr_raw = librosa.load(ref_audio_path, sr=None, mono=True)

    # Speaker encoder input (24 kHz)
    if sr_raw != spk_sr:
        wav_24k = librosa.resample(wav_raw, orig_sr=sr_raw, target_sr=spk_sr)
    else:
        wav_24k = wav_raw
    speaker_audio = torch.from_numpy(wav_24k.astype(np.float32)).unsqueeze(0).to(device)

    # Speech tokenizer input
    speech_tok = model.speech_tokenizer
    tok_sr = int(speech_tok.feature_extractor.sampling_rate)
    if sr_raw != tok_sr:
        wav_tok = librosa.resample(wav_raw, orig_sr=sr_raw, target_sr=tok_sr)
    else:
        wav_tok = wav_raw
    fe_out = speech_tok.feature_extractor(
        raw_audio=[wav_tok.astype(np.float32)],
        sampling_rate=tok_sr,
        return_tensors="pt",
    )
    tokenizer_audio = fe_out["input_values"].squeeze(1).to(device)
    tokenizer_padding_mask = fe_out["padding_mask"].squeeze(1).to(device)

    # ---- Extract speaker embedding ----------------------------------------
    with torch.no_grad():
        speaker_embedding = encoder._extract_speaker_embedding(speaker_audio)
        ref_audio_codes = encoder._encode_audio(tokenizer_audio, tokenizer_padding_mask)

    print(f"Speaker embedding: {speaker_embedding.shape}")
    print(f"Ref audio codes:   {ref_audio_codes.shape}  "
          f"({ref_audio_codes.shape[0]} frames × {ref_audio_codes.shape[1]} codebooks)")
    print(f"Ref text ids:      {ref_text_ids.shape}")
    print(f"Role prefix ids:   {role_prefix_ids.shape}")
    print(f"Codec header ids:  {codec_header_ids.shape}")

    return {
        "speaker_embedding": speaker_embedding.cpu().float(),
        "ref_audio_codes": ref_audio_codes.cpu(),
        "ref_text_ids": ref_text_ids.cpu(),
        "role_prefix_ids": role_prefix_ids.cpu(),
        "codec_header_ids": codec_header_ids.cpu(),
        "metadata": {
            "ref_text": ref_text,
            "ref_audio": ref_audio_path,
            "language": language,
            "model_path": model_path,
            "hidden_size": encoder.hidden_size,
            "num_code_groups": encoder.num_code_groups,
            "codec_pad_id": encoder.codec_pad_id,
            "codec_bos_id": encoder.codec_bos_id,
            "tts_bos_token_id": encoder.tts_bos_token_id,
            "tts_eos_token_id": encoder.tts_eos_token_id,
            "tts_pad_token_id": encoder.tts_pad_token_id,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference speaker data for Qwen3-TTS prefill encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python extract_reference.py \\
      --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
      --ref-text "I felt like, you know, you can be both, right?" \\
      --ref-audio scarlettjohansson.wav \\
      --output ref_data.pt
""",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="HuggingFace repo ID or local checkpoint directory.")
    parser.add_argument("--ref-text", required=True,
                        help="Transcript of the reference audio.")
    parser.add_argument("--ref-audio", required=True,
                        help="Path to reference audio file (wav).")
    parser.add_argument("--language", default="auto",
                        help="Target language (default: auto).")
    parser.add_argument("--output", required=True,
                        help="Path to save reference data (.pt).")
    parser.add_argument("--device", default="cuda:0",
                        help="Device (default: cuda:0).")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Dtype (default: bfloat16).")
    args = parser.parse_args()

    data = extract_reference(
        model_path=args.model_path,
        ref_text=args.ref_text,
        ref_audio_path=args.ref_audio,
        language=args.language,
        device=args.device,
        dtype_str=args.dtype,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(data, args.output)
    print(f"\nSaved reference data to {args.output}")


if __name__ == "__main__":
    main()
