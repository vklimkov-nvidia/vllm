"""
Triton Python Model for Qwen3-TTS end-to-end **streaming** inference.

Pipeline:
  1. Receive text (+ optional language) from Triton request
  2. Tokenize text with HuggingFace processor
  3. Run TorchScript PrefillAssembler to build dense prefill embeddings
  4. Run vLLM decode loop; every time codec_chunk_size frames accumulate,
     decode the chunk to audio via BLS (codec_decoder) and send it back
     as a partial Triton response.  The final partial response carries
     the TRITONSERVER_RESPONSE_COMPLETE_FINAL flag.
  5. Client concatenates received audio chunks.

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
  - codec_chunk_size          : max codec frames per TRT call (default: 128)
  - codec_left_context        : overlap frames between chunks (default: 25)
  - dtype                     : bfloat16 / float16 / float32 (default: bfloat16)
"""

import asyncio
import json
import logging
import os
import queue
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
    """Qwen3-TTS streaming Triton model (orchestrator).

    Inputs  (per request):
        text     : STRING [1]        — text to synthesize
        language : STRING [1]        — optional, codec language (e.g. "english", "auto")

    Outputs (streamed, multiple responses per request):
        audio    : FP32 [samples]    — audio chunk @ 24 kHz

    The model is **decoupled**: each request receives multiple partial
    responses (one per audio chunk) followed by a final response with
    the TRITONSERVER_RESPONSE_COMPLETE_FINAL flag.  Audio chunks are
    sent as soon as codec_chunk_size frames have been generated, so the
    client starts receiving audio well before the full utterance is done.
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
        self.codec_chunk_size = int(_get_param(params, "codec_chunk_size", "128"))
        self.codec_left_context = int(_get_param(params, "codec_left_context", "25"))
        self._samples_per_frame = int(24000 / 12.5)

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
    # Codec decode: tokens -> waveform via BLS to codec_decoder model
    # ------------------------------------------------------------------

    def _decode_codec_single(self, codec_tokens: torch.Tensor,
                             actual_frames: int) -> np.ndarray:
        """Decode a single chunk via BLS, padding to codec_chunk_size.

        All chunks are padded to a uniform size so the codec_decoder TRT
        engine always sees identical input shapes, enabling optimal batching.
        The output is trimmed back to actual_frames worth of audio.
        """
        num_q = codec_tokens.shape[1]
        pad_frames = self.codec_chunk_size - codec_tokens.shape[0]
        if pad_frames > 0:
            codec_tokens = torch.cat([
                codec_tokens,
                torch.zeros(pad_frames, num_q, dtype=codec_tokens.dtype),
            ], dim=0)

        codes_np = codec_tokens.cpu().numpy().astype(np.int64)
        codes_np = np.expand_dims(codes_np, axis=0)  # [T, Q] -> [1, T, Q]

        input_tensor = pb_utils.Tensor("audio_codes", codes_np)
        request = pb_utils.InferenceRequest(
            model_name="codec_decoder",
            requested_output_names=["audio_values"],
            inputs=[input_tensor],
        )
        response = request.exec()

        if response.has_error():
            raise RuntimeError(f"Codec decode failed: {response.error().message()}")

        audio_tensor = pb_utils.get_output_tensor_by_name(
            response, "audio_values"
        )
        if audio_tensor.is_cpu():
            audio = audio_tensor.as_numpy()
        else:
            audio = torch.from_dlpack(audio_tensor.to_dlpack()).cpu().numpy()
        if audio.ndim > 1:
            audio = audio[0]

        expected_samples = actual_frames * self._samples_per_frame
        return audio[:expected_samples]

    def _send_audio_chunk(self, response_sender, audio: np.ndarray, final: bool):
        out = pb_utils.Tensor("audio", audio.astype(np.float32))
        flags = (pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0)
        response_sender.send(
            pb_utils.InferenceResponse(output_tensors=[out]),
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Codec worker thread: reads (chunk_tokens, ctx) from a queue,
    # decodes via BLS, and sends audio to the client.  Runs in its
    # own thread so the vLLM decode loop is never blocked by TRT.
    # ------------------------------------------------------------------

    def _codec_worker(self, codec_q, response_sender, state):
        """Drains *codec_q* until a sentinel is received.

        Each item is a ``(chunk_tokens, ctx_frames, is_final)`` tuple.
        The chunk is decoded via BLS and streamed back as a partial
        response.  The last real chunk carries ``COMPLETE_FINAL`` so no
        empty trailing response is needed.

        **All** sends happen on this thread so that Triton sees a
        single-threaded FIFO stream of responses.

        *state* is a dict written by this thread and read (after join)
        by the caller for timing / error propagation.
        """
        finalized = False
        try:
            while True:
                item = codec_q.get()
                if item is None:
                    self._send_audio_chunk(
                        response_sender,
                        np.array([], dtype=np.float32),
                        final=True,
                    )
                    finalized = True
                    break
                chunk_tokens, ctx, is_final = item

                t0 = time.perf_counter()
                audio = self._decode_codec_single(chunk_tokens,
                                                  chunk_tokens.shape[0])
                t1 = time.perf_counter()

                trim = ctx * self._samples_per_frame
                audio = audio[trim:]

                self._send_audio_chunk(response_sender, audio, final=is_final)
                finalized = is_final

                state["chunks_sent"] += 1
                state["total_samples"] += len(audio)
                state["codec_decode_ms"] += (t1 - t0) * 1000
                if state["t_first_audio"] is None:
                    state["t_first_audio"] = time.perf_counter()

                if is_final:
                    return
        except Exception as e:
            state["error"] = e
            if not finalized:
                try:
                    response_sender.send(
                        pb_utils.InferenceResponse(output_tensors=[]),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Streaming pipeline: the decode thread generates codec tokens at
    # full speed and feeds chunks to the codec worker via a Queue.
    # ------------------------------------------------------------------

    def _synthesize_streaming(self, text: str, language: str, response_sender):
        t_start = time.perf_counter()

        prefill = self._build_prefill(text, language)
        prefill_emb = prefill[0].contiguous().cpu()
        t_prefill_done = time.perf_counter()

        request_id = str(uuid.uuid4())
        prompt_len = prefill_emb.shape[0]

        inputs = {
            "prompt_token_ids": [0] * prompt_len,
            "custom_inputs": {"combined_embeddings": prefill_emb},
        }

        output_queue = asyncio.run_coroutine_threadsafe(
            self.engine.add_request(request_id, inputs, self.sampling_params),
            self._loop,
        ).result(timeout=30)

        prefill_output = asyncio.run_coroutine_threadsafe(
            output_queue.get(), self._loop,
        ).result(timeout=60)

        next_input = prefill_output.outputs[0].custom_outputs[
            "next_input_embeddings"
        ][-1:, :]
        first_token = prefill_output.outputs[0].custom_outputs["codes"][-1:]

        generated_codecs = [first_token]

        chunk_size = self.codec_chunk_size
        left_context = self.codec_left_context
        sent_frames = 0

        codec_q: queue.Queue = queue.Queue()
        state = {
            "t_first_audio": None,
            "total_samples": 0,
            "chunks_sent": 0,
            "codec_decode_ms": 0.0,
            "error": None,
        }
        codec_thread = threading.Thread(
            target=self._codec_worker,
            args=(codec_q, response_sender, state),
            daemon=True,
        )
        codec_thread.start()

        t_decode_start = time.perf_counter()
        decode_step_times = []

        for step in range(self.max_tokens - 1):
            if state["error"] is not None:
                break

            t_step = time.perf_counter()
            outputs = self.engine.decode_step_shm(
                request_id,
                custom_inputs={"combined_embeddings": next_input},
            )
            decode_step_times.append(time.perf_counter() - t_step)

            next_input = outputs["next_input_embeddings"][-1:, :]
            next_tokens = outputs["codes"][-1:].clone()

            if next_tokens[0, 0].item() == self.codec_eos_token_id:
                break

            generated_codecs.append(next_tokens)
            total_frames = len(generated_codecs)

            if sent_frames == 0:
                ctx = 0
                needed = chunk_size
            else:
                ctx = left_context
                needed = chunk_size - left_context

            if total_frames - sent_frames >= needed:
                chunk = torch.cat(
                    generated_codecs[sent_frames - ctx : sent_frames + needed],
                    dim=0,
                )
                codec_q.put((chunk, ctx, False))
                sent_frames += needed

        t_decode_end = time.perf_counter()

        total_frames = len(generated_codecs)
        remaining = total_frames - sent_frames

        if remaining > 0:
            ctx = left_context if sent_frames > 0 else 0
            chunk = torch.cat(
                generated_codecs[sent_frames - ctx :], dim=0,
            )
            codec_q.put((chunk, ctx, True))
        else:
            codec_q.put(None)

        t_abort_start = time.perf_counter()
        asyncio.run_coroutine_threadsafe(
            self.engine.abort(request_id), self._loop,
        ).result(timeout=10)
        t_abort_end = time.perf_counter()

        codec_thread.join(timeout=30)

        if state["error"] is not None:
            raise state["error"]

        t_first_audio = state["t_first_audio"] or time.perf_counter()
        t_end = time.perf_counter()
        avg_step = (sum(decode_step_times) / len(decode_step_times) * 1000
                    if decode_step_times else 0)
        logger.info(
            "[Streaming] prefill: %.1fms | vllm_prefill: %.1fms | "
            "decode_loop: %.1fms (%d steps, avg %.2fms/step) | "
            "abort: %.1fms | codec_bls: %.1fms (%d chunks) | "
            "TTFA: %.1fms | total: %.1fms | "
            "frames: %d | audio: %.2fs",
            (t_prefill_done - t_start) * 1000,
            (t_decode_start - t_prefill_done) * 1000,
            (t_decode_end - t_decode_start) * 1000,
            len(decode_step_times), avg_step,
            (t_abort_end - t_abort_start) * 1000,
            state["codec_decode_ms"], state["chunks_sent"],
            (t_first_audio - t_start) * 1000,
            (t_end - t_start) * 1000,
            total_frames,
            state["total_samples"] / 24000.0,
        )

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
            self._synthesize_streaming(text, language, response_sender)
        except Exception as e:
            logger.error("Request failed: %s", e, exc_info=True)
            try:
                response_sender.send(
                    pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError(str(e)),
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
            except Exception:
                pass

    def finalize(self):
        logger.info("Shutting down Qwen3-TTS TritonPythonModel")
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=10)
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False)
