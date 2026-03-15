"""
Triton Python Model for Qwen3-TTS end-to-end inference.

Pipeline:
  1. Receive text (+ optional language) from Triton request
  2. Tokenize text with HuggingFace processor
  3. Run TorchScript PrefillAssembler to build dense prefill embeddings
  4. Submit to vLLM engine for autoregressive codec token generation
  5. Decode codec tokens to waveform via TensorRT codec engine
  6. Return audio waveform as Triton response

Required artifacts in the model directory:
  - reference.pt              : pre-extracted speaker data (speaker_embedding,
                                 ref_audio_codes, ref_text_ids, role_prefix_ids)
  - prefill_assembler.pt      : TorchScript-exported PrefillAssembler
  - vllm_model/               : vLLM-compatible Qwen3-TTS talker checkpoint
  - codec_decoder.plan        : TensorRT engine for codec decoder (ONNX->TRT)

Model config parameters (set in config.pbtxt):
  - vllm_model_path           : path to vLLM model dir      (default: vllm_model)
  - reference_path            : path to reference.pt         (default: reference.pt)
  - prefill_assembler_path    : path to TorchScript .pt      (default: prefill_assembler.pt)
  - codec_engine_path         : path to TRT .plan engine     (default: codec_decoder.plan)
  - max_tokens                : max decode steps              (default: 2048)
  - dtype                     : bfloat16 / float16 / float32 (default: bfloat16)
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

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
logger = logging.getLogger("qwen3_tts_triton")


def _get_param(parameters: dict, key: str, default: str) -> str:
    val = parameters.get(key)
    if val is None:
        return default
    if isinstance(val, dict):
        return val.get("string_value", default)
    return str(val)


class TritonPythonModel:
    """Qwen3-TTS end-to-end Triton model.

    Inputs  (per request):
        text     : STRING [1]        — text to synthesize
        language : STRING [1]        — optional, codec language (e.g. "english", "auto")

    Outputs (per request):
        audio    : FP32 [samples]    — synthesized waveform @ 24 kHz
    """

    # ------------------------------------------------------------------
    # Triton lifecycle
    # ------------------------------------------------------------------

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        model_dir = Path(os.path.dirname(__file__))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dtype_str = _get_param(params, "dtype", "bfloat16")
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_str]

        self.max_tokens = int(_get_param(params, "max_tokens", "2048"))

        self._load_tokenizer(params, model_dir)
        self._load_prefill_assembler(params, model_dir)
        self._load_reference_data(params, model_dir)
        self._load_codec_engine(params, model_dir)
        self._init_vllm_engine(params, model_dir)

        logger.info("Qwen3-TTS TritonPythonModel initialized successfully")

    def _load_tokenizer(self, params: dict, model_dir: Path):
        from transformers import AutoTokenizer

        vllm_model = _get_param(params, "vllm_model_path", str(model_dir / "vllm_model"))
        self.tokenizer = AutoTokenizer.from_pretrained(vllm_model, trust_remote_code=True)
        logger.info("Loaded tokenizer from %s", vllm_model)

    def _load_prefill_assembler(self, params: dict, model_dir: Path):
        pa_path = model_dir / _get_param(params, "prefill_assembler_path", "prefill_assembler.pt")
        self.prefill_assembler = torch.jit.load(str(pa_path), map_location=self.device)
        self.prefill_assembler.eval()
        logger.info("Loaded TorchScript PrefillAssembler from %s", pa_path)

    def _load_reference_data(self, params: dict, model_dir: Path):
        ref_path = model_dir / _get_param(params, "reference_path", "reference.pt")
        ref = torch.load(str(ref_path), map_location="cpu", weights_only=False)
        self.speaker_embedding = ref["speaker_embedding"].unsqueeze(0).to(self.device, self.dtype)
        self.ref_audio_codes = ref["ref_audio_codes"].to(self.device, torch.long)
        self.ref_text_ids = ref["ref_text_ids"].to(self.device, torch.long)
        self.role_prefix_ids = ref["role_prefix_ids"]

        self.codec_language_mapping: Optional[Dict[str, int]] = ref.get("codec_language_mapping")
        self.codec_eos_token_id = int(ref.get("codec_eos_token_id", 2150))
        logger.info("Loaded reference data from %s", ref_path)

    def _load_codec_engine(self, params: dict, model_dir: Path):
        engine_path = Path(
            _get_param(params, "codec_engine_path", str(model_dir / "codec_decoder.plan"))
        )
        if not engine_path.exists():
            raise FileNotFoundError(f"Codec TRT engine not found at {engine_path}")

        with open(engine_path, "rb") as f:
            engine_buffer = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        self._codec_engine = runtime.deserialize_cuda_engine(engine_buffer)
        self._codec_context = self._codec_engine.create_execution_context()
        self._codec_stream = torch.cuda.Stream(device=self.device)
        logger.info("Loaded TensorRT codec engine from %s", engine_path)

        # hardcode the samples per frame.
        # 12.5 frames per second at 24khz
        self._samples_per_frame = int(24000 / 12.5)

    def _init_vllm_engine(self, params: dict, model_dir: Path):
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        vllm_model = _get_param(params, "vllm_model_path", str(model_dir / "vllm_model"))
        dtype_str = _get_param(params, "dtype", "bfloat16")

        engine_args = AsyncEngineArgs(
            model=str(Path(vllm_model).absolute()),
            dtype=dtype_str,
            max_model_len=self.max_tokens,
            gpu_memory_utilization=float(_get_param(params, "gpu_memory_utilization", "0.6")),
            skip_tokenizer_init=True,
            enable_prefix_caching=False,
            trust_remote_code=True,
            input_coalesce_timeout_ms=5,
            compilation_config={"cudagraph_mode": "PIECEWISE"},
            shm_decode=True,
        )

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self.engine = AsyncLLM.from_engine_args(engine_args)

        # AsyncLLM.add_request() does NOT start the output_handler
        # (only generate() does). We must kick it off manually on our
        # event loop so engine-core results are dispatched to request queues.
        asyncio.run_coroutine_threadsafe(
            self._start_output_handler(), self._loop
        ).result(timeout=10)

        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            skip_sampling=False,
        )

        self._thread_pool = __import__("concurrent").futures.ThreadPoolExecutor(
            max_workers=int(_get_param(params, "max_concurrency", "8")),
        )

        logger.info("vLLM engine initialized with model %s", vllm_model)

    async def _start_output_handler(self):
        """Must be called from within the running event loop so that
        asyncio.create_task() inside _run_output_handler() picks up
        the correct loop."""
        self.engine._run_output_handler()

    # ------------------------------------------------------------------
    # Text tokenization (mirrors export_prefill_encoder.tokenize_target_text)
    # ------------------------------------------------------------------

    def _tokenize_text(self, text: str) -> torch.Tensor:
        synth_full = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer(synth_full, return_tensors="pt", padding=True)["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids[:, 3:-5].to(torch.long)

    # ------------------------------------------------------------------
    # Resolve language -> codec language id
    # ------------------------------------------------------------------

    def _resolve_language_id(self, language: str) -> Optional[torch.Tensor]:
        key = language.strip().lower()
        if key == "auto":
            return None
        if self.codec_language_mapping and key in self.codec_language_mapping:
            lang_id = self.codec_language_mapping[key]
            return torch.tensor([lang_id], device=self.device, dtype=torch.long)
        return None

    # ------------------------------------------------------------------
    # Prefill: text -> dense embeddings
    # ------------------------------------------------------------------

    def _build_prefill(self, text: str, language: str) -> torch.Tensor:
        text_ids = self._tokenize_text(text).to(self.device, torch.long)
        language_id = self._resolve_language_id(language)

        with torch.inference_mode():
            prefill = self.prefill_assembler(
                self.speaker_embedding,
                self.ref_audio_codes,
                text_ids,
                self.ref_text_ids,
                language_id,
            )
        return prefill  # [1, S, D]

    # ------------------------------------------------------------------
    # vLLM generation: prefill_emb -> codec tokens
    # ------------------------------------------------------------------

    async def _generate_codec_tokens(self, prefill_emb: torch.Tensor) -> torch.Tensor:
        """Run vLLM generation loop and collect codec tokens."""
        t_total_start = time.perf_counter()

        request_id = str(uuid.uuid4())
        prompt_len = prefill_emb.shape[0]

        inputs = {
            "prompt_token_ids": [0] * prompt_len,
            "custom_inputs": {
                "combined_embeddings": prefill_emb,
            },
        }

        t_prefill_start = time.perf_counter()
        queue = await self.engine.add_request(request_id, inputs, self.sampling_params)
        prefill_output = await queue.get()
        t_prefill_end = time.perf_counter()

        next_input = prefill_output.outputs[0].custom_outputs["next_input_embeddings"][-1:, :]
        next_tokens = prefill_output.outputs[0].custom_outputs["codes"][-1:]

        generated_codecs = [next_tokens]

        decode_times = []
        for step in range(self.max_tokens - 1):
            t_step_start = time.perf_counter()
            outputs = self.engine.decode_step_shm(
                request_id,
                custom_inputs={"combined_embeddings": next_input},
            )
            t_step_end = time.perf_counter()
            decode_times.append(t_step_end - t_step_start)

            next_input = outputs["next_input_embeddings"][-1:, :]
            next_tokens = outputs["codes"][-1:].clone()

            if next_tokens[0, 0].item() == self.codec_eos_token_id:
                break

            generated_codecs.append(next_tokens)

        await self.engine.abort(request_id)
        t_total_end = time.perf_counter()

        avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
        print(f"[vLLM timing] prefill: {(t_prefill_end - t_prefill_start)*1000:.1f}ms | "
              f"decode steps: {len(decode_times)} | avg decode_step_shm: {avg_decode*1000:.2f}ms | "
              f"total generation: {(t_total_end - t_total_start)*1000:.1f}ms", flush=True)

        return torch.cat(generated_codecs, dim=0)  # [T, num_code_groups]

    # ------------------------------------------------------------------
    # Codec decode: tokens -> waveform
    # ------------------------------------------------------------------

    def _decode_codec(self, codec_tokens: torch.Tensor) -> np.ndarray:
        """Decode codec tokens [T, Q] into waveform via the TensorRT codec engine.

        The TRT engine expects input ``audio_codes`` of shape [B, T, Q] (int64)
        and produces ``audio_values`` of shape [B, samples] (float32).
        """
        num_frames = codec_tokens.shape[0]
        codes = codec_tokens.unsqueeze(0).to(self.device, dtype=torch.int64).contiguous()
        input_shape = tuple(codes.shape)

        ctx = self._codec_context
        ctx.set_input_shape("audio_codes", input_shape)

        output_shape = tuple(ctx.get_tensor_shape("audio_values"))
        audio_out = torch.empty(output_shape, dtype=torch.float32, device=self.device)

        print(f"TRT codec: input_shape={input_shape}, output_shape={output_shape}", flush=True)

        ctx.set_tensor_address("audio_codes", codes.data_ptr())
        ctx.set_tensor_address("audio_values", audio_out.data_ptr())

        with torch.cuda.stream(self._codec_stream):
            ok = ctx.execute_async_v3(self._codec_stream.cuda_stream)
        self._codec_stream.synchronize()

        if not ok:
            raise RuntimeError("TensorRT codec decoder execution failed")

        waveform = audio_out.squeeze(0).cpu().numpy()

        # TRT may return a buffer sized for the optimization profile rather
        # than the actual input.  Trim to the expected sample count.
        expected_samples = num_frames * self._samples_per_frame
        if waveform.shape[0] > expected_samples:
            print(f"Trimming TRT output from {waveform.shape[0]} to {expected_samples} samples", flush=True)
            waveform = waveform[:expected_samples]

        return waveform

    # ------------------------------------------------------------------
    # Full pipeline: text -> audio
    # ------------------------------------------------------------------

    def _synthesize(self, text: str, language: str) -> np.ndarray:
        """Run the full TTS pipeline synchronously (called from a thread)."""
        t_synth_start = time.perf_counter()

        t_prefill_build_start = time.perf_counter()
        prefill = self._build_prefill(text, language)
        prefill_emb = prefill[0].contiguous().cpu()
        t_prefill_build_end = time.perf_counter()

        future = asyncio.run_coroutine_threadsafe(
            self._generate_codec_tokens(prefill_emb), self._loop
        )
        codec_tokens = future.result(timeout=600)
        print(f"Codec tokens to decode: {codec_tokens.shape[0]} frames x {codec_tokens.shape[1]} quantizers", flush=True)

        t_codec_start = time.perf_counter()
        audio = self._decode_codec(codec_tokens)
        t_codec_end = time.perf_counter()

        t_synth_end = time.perf_counter()
        print(f"[Pipeline timing] prefill_build: {(t_prefill_build_end - t_prefill_build_start)*1000:.1f}ms | "
              f"codec decode: {(t_codec_end - t_codec_start)*1000:.1f}ms | "
              f"total synthesis: {(t_synth_end - t_synth_start)*1000:.1f}ms", flush=True)

        return audio

    # ------------------------------------------------------------------
    # Triton execute
    # ------------------------------------------------------------------

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                text = text_tensor.as_numpy()[0].decode("utf-8")

                lang_tensor = pb_utils.get_input_tensor_by_name(request, "language")
                if lang_tensor is not None:
                    language = lang_tensor.as_numpy()[0].decode("utf-8")
                else:
                    language = "auto"

                logger.info("Synthesizing text=%r language=%s", text[:80], language)
                audio = self._synthesize(text, language)
                logger.info("Generated %d audio samples (%.2f s @ 24 kHz)", len(audio), len(audio) / 24000.0)

                out_tensor = pb_utils.Tensor("audio", audio)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

            except Exception as e:
                logger.error("Request failed: %s", e, exc_info=True)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(e)),
                    )
                )

        return responses

    def finalize(self):
        logger.info("Shutting down Qwen3-TTS TritonPythonModel")
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=10)
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False)
        if hasattr(self, "_codec_context"):
            del self._codec_context
        if hasattr(self, "_codec_engine"):
            del self._codec_engine
