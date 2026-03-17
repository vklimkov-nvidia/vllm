"""
Triton Python Model for Qwen3-TTS end-to-end inference.

Pipeline:
  1. Receive text (+ optional language) from Triton request
  2. Tokenize text with HuggingFace processor
  3. Run TorchScript PrefillAssembler to build dense prefill embeddings
  4. Submit to vLLM engine for autoregressive codec token generation
  5. Decode codec tokens to waveform via BLS call to the codec_decoder model
     (Triton's dynamic batcher batches these calls for efficient TRT inference)
  6. Return audio waveform as Triton response

Required artifacts in the model directory:
  - reference.pt              : pre-extracted speaker data (speaker_embedding,
                                 ref_audio_codes, ref_text_ids, role_prefix_ids)
  - prefill_assembler.pt      : TorchScript-exported PrefillAssembler
  - vllm_model/               : vLLM-compatible Qwen3-TTS talker checkpoint

Companion Triton model:
  - codec_decoder             : TRT-backed model for batched codec→waveform
                                 decoding (deployed separately in model_repository)

Model config parameters (set in config.pbtxt):
  - vllm_model_path           : path to vLLM model dir      (default: vllm_model)
  - reference_path            : path to reference.pt         (default: reference.pt)
  - prefill_assembler_path    : path to TorchScript .pt      (default: prefill_assembler.pt)
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
import torch
import triton_python_backend_utils as pb_utils

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
    """Qwen3-TTS end-to-end Triton model (orchestrator).

    Inputs  (per request):
        text     : STRING [1]        — text to synthesize
        language : STRING [1]        — optional, codec language (e.g. "english", "auto")

    Outputs (per request):
        audio    : FP32 [samples]    — synthesized waveform @ 24 kHz

    With max_batch_size > 0 and dynamic_batching enabled, Triton collects
    up to N requests before calling execute().  Each request is processed
    concurrently via a thread pool:
      - PrefillAssembler runs in-thread (lightweight, GPU, thread-safe)
      - vLLM generation runs a synchronous decode loop per thread; the
        engine batches across active requests internally
      - Codec decoding is dispatched via BLS to the ``codec_decoder`` model,
        whose own dynamic batcher groups concurrent decode requests into
        efficient TRT batches
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
            input_coalesce_timeout_ms=30,
            compilation_config={"cudagraph_mode": "PIECEWISE"},
            shm_decode=True,
        )

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self.engine = AsyncLLM.from_engine_args(engine_args)

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
    #
    # This is a *synchronous* method designed to run in a thread-pool
    # thread.  The async parts (add_request, queue.get, abort) are
    # dispatched to the dedicated event loop via run_coroutine_threadsafe.
    # The decode loop itself runs synchronously in the calling thread,
    # which allows multiple threads to drive their own decode loops
    # concurrently — vLLM's engine batches across all active requests.
    # ------------------------------------------------------------------

    def _generate_codec_tokens(self, prefill_emb: torch.Tensor) -> torch.Tensor:
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
        queue = asyncio.run_coroutine_threadsafe(
            self.engine.add_request(request_id, inputs, self.sampling_params),
            self._loop,
        ).result(timeout=30)

        prefill_output = asyncio.run_coroutine_threadsafe(
            queue.get(), self._loop,
        ).result(timeout=60)
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

        asyncio.run_coroutine_threadsafe(
            self.engine.abort(request_id), self._loop,
        ).result(timeout=10)

        t_total_end = time.perf_counter()

        avg_decode = sum(decode_times) / len(decode_times) if decode_times else 0
        print(f"[vLLM timing] prefill: {(t_prefill_end - t_prefill_start)*1000:.1f}ms | "
              f"decode steps: {len(decode_times)} | avg decode_step_shm: {avg_decode*1000:.2f}ms | "
              f"total generation: {(t_total_end - t_total_start)*1000:.1f}ms", flush=True)

        return torch.cat(generated_codecs, dim=0)  # [T, num_code_groups]

    # ------------------------------------------------------------------
    # Codec decode: tokens -> waveform via BLS to codec_decoder model
    # ------------------------------------------------------------------

    def _decode_codec_bls(self, codec_tokens: torch.Tensor) -> np.ndarray:
        """Send codec tokens to the codec_decoder Triton model via BLS.

        Triton's dynamic batcher on codec_decoder collects concurrent BLS
        requests from multiple pipeline threads into efficient TRT batches.
        """
        codes_np = codec_tokens.cpu().numpy().astype(np.int64)
        codes_np = np.expand_dims(codes_np, axis=0)  # [T, Q] -> [1, T, Q] batch dim

        input_tensor = pb_utils.Tensor("audio_codes", codes_np)
        request = pb_utils.InferenceRequest(
            model_name="codec_decoder",
            requested_output_names=["audio_values"],
            inputs=[input_tensor],
        )
        response = request.exec()

        if response.has_error():
            raise RuntimeError(f"Codec decode failed: {response.error().message()}")

        audio = pb_utils.get_output_tensor_by_name(
            response, "audio_values"
        ).as_numpy()
        if audio.ndim > 1:
            audio = audio[0]  # strip batch dim from codec_decoder response
        return audio

    # ------------------------------------------------------------------
    # Full pipeline: text -> audio  (runs in a thread-pool thread)
    # ------------------------------------------------------------------

    def _synthesize(self, text: str, language: str) -> np.ndarray:
        t_synth_start = time.perf_counter()

        t_prefill_build_start = time.perf_counter()
        prefill = self._build_prefill(text, language)
        prefill_emb = prefill[0].contiguous().cpu()
        t_prefill_build_end = time.perf_counter()

        codec_tokens = self._generate_codec_tokens(prefill_emb)
        print(f"Codec tokens to decode: {codec_tokens.shape[0]} frames x {codec_tokens.shape[1]} quantizers", flush=True)

        t_codec_start = time.perf_counter()
        audio = self._decode_codec_bls(codec_tokens)
        t_codec_end = time.perf_counter()

        t_synth_end = time.perf_counter()
        print(f"[Pipeline timing] prefill_build: {(t_prefill_build_end - t_prefill_build_start)*1000:.1f}ms | "
              f"codec decode: {(t_codec_end - t_codec_start)*1000:.1f}ms | "
              f"total synthesis: {(t_synth_end - t_synth_start)*1000:.1f}ms", flush=True)

        return audio

    # ------------------------------------------------------------------
    # Triton execute — concurrent via thread pool
    # ------------------------------------------------------------------

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                text = text_tensor.as_numpy().flatten()[0].decode("utf-8")

                lang_tensor = pb_utils.get_input_tensor_by_name(request, "language")
                if lang_tensor is not None:
                    language = lang_tensor.as_numpy().flatten()[0].decode("utf-8")
                else:
                    language = "auto"

                logger.info("Synthesizing text=%r language=%s", text[:80], language)
                self._thread_pool.submit(
                    self._handle_request, text, language, response_sender,
                )

            except Exception as e:
                logger.error("Request parse failed: %s", e, exc_info=True)
                response_sender.send(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(e)),
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

        return None

    def _handle_request(self, text: str, language: str, response_sender):
        try:
            audio = self._synthesize(text, language)
            logger.info("Generated %d audio samples (%.2f s @ 24 kHz)", len(audio), len(audio) / 24000.0)
            out_tensor = pb_utils.Tensor("audio", audio)
            response_sender.send(
                pb_utils.InferenceResponse(output_tensors=[out_tensor]),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
        except Exception as e:
            logger.error("Request failed: %s", e, exc_info=True)
            response_sender.send(
                pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(str(e)),
                ),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )

    def finalize(self):
        logger.info("Shutting down Qwen3-TTS TritonPythonModel")
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=10)
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False)
