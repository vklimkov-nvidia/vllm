"""
Triton Python model for batched codec decoding via TensorRT.

Triton's dynamic batcher collects individual decode requests into batches.
This model pads them to a uniform frame count, runs the TRT engine once,
and trims each output to the original length.

Input:
    audio_codes  : INT64 [frames, num_quantizers]  — codec tokens

Output:
    audio_values : FP32 [samples]                  — decoded waveform @ 24 kHz
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import triton_python_backend_utils as pb_utils

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("codec_decoder")


def _get_param(parameters: dict, key: str, default: str) -> str:
    val = parameters.get(key)
    if val is None:
        return default
    if isinstance(val, dict):
        return val.get("string_value", default)
    return str(val)


class TritonPythonModel:
    """Batched TensorRT codec decoder for Qwen3-TTS.

    Triton's dynamic batcher collects individual decode requests.
    execute() receives up to max_batch_size requests, pads variable-length
    codec token sequences to the longest in the batch, runs the TRT engine
    once, and trims each waveform output to its original frame count.
    """

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        engine_path = Path(
            _get_param(params, "codec_engine_path", "codec_decoder.plan")
        )
        if not engine_path.exists():
            raise FileNotFoundError(f"Codec TRT engine not found at {engine_path}")

        with open(engine_path, "rb") as f:
            engine_buffer = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self._engine = runtime.deserialize_cuda_engine(engine_buffer)
        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream(device=self.device)

        # 12.5 frames/s at 24 kHz
        self._samples_per_frame = int(24000 / 12.5)

        logger.info("Codec decoder initialized from %s", engine_path)

    def execute(self, requests):
        t_start = time.perf_counter()
        batch_size = len(requests)

        codes_list = []
        frame_counts = []
        for req in requests:
            tensor = pb_utils.get_input_tensor_by_name(req, "audio_codes")
            codes = torch.from_numpy(tensor.as_numpy())
            if codes.dim() == 3:
                codes = codes.squeeze(0)  # strip batch dim [1, T, Q] -> [T, Q]
            codes_list.append(codes)
            frame_counts.append(codes.shape[0])

        max_frames = max(frame_counts)
        num_q = codes_list[0].shape[1]

        padded = torch.zeros(batch_size, max_frames, num_q, dtype=torch.int64)
        for i, codes in enumerate(codes_list):
            padded[i, : codes.shape[0], :] = codes
        padded = padded.to(self.device).contiguous()

        input_shape = tuple(padded.shape)
        ctx = self._context
        ctx.set_input_shape("audio_codes", input_shape)

        output_shape = tuple(ctx.get_tensor_shape("audio_values"))
        audio_out = torch.empty(
            output_shape, dtype=torch.float32, device=self.device
        )

        ctx.set_tensor_address("audio_codes", padded.data_ptr())
        ctx.set_tensor_address("audio_values", audio_out.data_ptr())

        with torch.cuda.stream(self._stream):
            ok = ctx.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        if not ok:
            error = pb_utils.TritonError("TensorRT codec execution failed")
            return [
                pb_utils.InferenceResponse(output_tensors=[], error=error)
                for _ in requests
            ]

        t_end = time.perf_counter()
        logger.info(
            "Codec batch: %d requests, max_frames=%d, %.1fms",
            batch_size,
            max_frames,
            (t_end - t_start) * 1000,
        )

        responses = []
        for i in range(batch_size):
            expected = frame_counts[i] * self._samples_per_frame
            waveform = audio_out[i, :expected].cpu().numpy()
            out = pb_utils.Tensor("audio_values", waveform)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))

        return responses

    def finalize(self):
        if hasattr(self, "_context"):
            del self._context
        if hasattr(self, "_engine"):
            del self._engine
        logger.info("Codec decoder finalized")
