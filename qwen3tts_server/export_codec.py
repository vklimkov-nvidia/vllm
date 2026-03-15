#!/usr/bin/env python3
"""
Export Qwen3-TTS 12Hz codec decoder to ONNX (and optionally TensorRT).

Input:  audio_codes  [batch, frames, num_quantizers]  int64
Output: audio_values [batch, samples]                 float32
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Compatibility patches — must run BEFORE qwen_tts is imported.
#
# 1) check_model_inputs: removed in transformers >=5.x but the Qwen3-TTS
#    modeling code imports it at module level.
# 2) create_causal_mask / create_sliding_window_causal_mask: use torch.vmap
#    which is untraceable by ONNX export.  Returning None makes attention
#    layers fall back to their built-in causal behaviour.
# ---------------------------------------------------------------------------
import transformers.utils.generic as _tg
if not hasattr(_tg, "check_model_inputs"):
    _tg.check_model_inputs = lambda func=None, *a, **kw: (
        func if callable(func) else (lambda fn: fn)
    )

try:
    import transformers.masking_utils as _mu
    for _name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if hasattr(_mu, _name):
            setattr(_mu, _name, lambda *a, **kw: None)
except ImportError:
    pass

# scaled_dot_product_attention with enable_gqa=True is not supported by the
# TorchScript ONNX exporter.  Since num_heads == num_kv_heads in the decoder,
# GQA is a no-op — just strip the kwarg so tracing succeeds.
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention

def _sdpa_no_gqa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_sdpa(*args, **kwargs)

torch.nn.functional.scaled_dot_product_attention = _sdpa_no_gqa

from qwen_tts import Qwen3TTSTokenizer  # noqa: E402


class CodecDecoderWrapper(torch.nn.Module):
    """Thin wrapper: transposes [B,T,Q] → [B,Q,T] for the decoder."""

    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(audio_codes.transpose(1, 2)).squeeze(1)


def check_onnx_parity(wrapper, onnx_path, audio_codes, device, atol=1e-3):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed – skipping parity check")
        return

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        and "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=providers)

    with torch.inference_mode():
        ref = wrapper(audio_codes).detach().cpu().float().numpy()
    ort_out = sess.run(None, {"audio_codes": audio_codes.cpu().numpy()})[0]

    max_diff = float(np.abs(ref - ort_out).max())
    ok = max_diff <= atol
    print(f"ONNX parity: max_abs_diff={max_diff:.6f}  atol={atol}  {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise RuntimeError("Parity check failed")


def convert_to_trt(onnx_path, trt_path, trtexec_bin, nq, batch_prof, frames_prof, fp16):
    exe = shutil.which(trtexec_bin) if "/" not in trtexec_bin else trtexec_bin
    if exe is None:
        raise FileNotFoundError(f"trtexec not found: {trtexec_bin}")
    trt_path.parent.mkdir(parents=True, exist_ok=True)

    def s(b, f):
        return f"{b}x{f}x{nq}"

    cmd = [
        exe,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        f"--minShapes=audio_codes:{s(batch_prof[0], frames_prof[0])}",
        f"--optShapes=audio_codes:{s(batch_prof[1], frames_prof[1])}",
        f"--maxShapes=audio_codes:{s(batch_prof[2], frames_prof[2])}",
    ]
    if fp16:
        cmd.append("--fp16")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"TensorRT engine saved to {trt_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Export Qwen3-TTS 12Hz codec decoder")
    p.add_argument("--tokenizer-path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--onnx-path", default="codec_decoder_12hz.onnx")
    p.add_argument("--frames", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--trt-path", default=None)
    p.add_argument("--trtexec-bin", default="/usr/src/tensorrt/bin/trtexec")
    p.add_argument("--trt-batch-profile", nargs=3, type=int, default=[1, 4, 32],
                   metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--trt-frames-profile", nargs=3, type=int, default=[1, 128, 1024],
                   metavar=("MIN", "OPT", "MAX"))
    p.add_argument("--trt-fp16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_path,
        device_map=args.device,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    decoder = tokenizer.model.decoder
    wrapper = CodecDecoderWrapper(decoder).to(device).eval()

    nq = int(decoder.config.num_quantizers)
    dummy = torch.randint(
        0, int(decoder.config.codebook_size),
        (args.batch_size, args.frames, nq),
        dtype=torch.long, device=device,
    )

    onnx_path = Path(args.onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper, (dummy,), str(onnx_path),
            dynamo=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["audio_codes"],
            output_names=["audio_values"],
            dynamic_axes={
                "audio_codes": {0: "batch", 1: "frames"},
                "audio_values": {0: "batch", 1: "samples"},
            },
        )
    print(f"ONNX exported to {onnx_path}")
    check_onnx_parity(wrapper, onnx_path, dummy, device)

    if args.trt_path:
        convert_to_trt(
            onnx_path, Path(args.trt_path), args.trtexec_bin,
            nq=nq,
            batch_prof=tuple(args.trt_batch_profile),
            frames_prof=tuple(args.trt_frames_profile),
            fp16=args.trt_fp16,
        )


if __name__ == "__main__":
    main()
