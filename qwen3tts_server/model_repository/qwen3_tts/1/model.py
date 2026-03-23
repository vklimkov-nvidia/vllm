"""
Triton Python Model for Qwen3-TTS CustomVoice **streaming** inference.

Pipeline:
  1. Receive text (+ optional language) from Triton request
  2. Tokenize text with HuggingFace tokenizer
  3. Run TorchScript PrefillAssembler to build dense prefill embeddings
     (speaker identity is a learned embedding — no reference audio)
  4. Run vLLM decode loop; every time codec_chunk_size frames accumulate,
     decode the chunk to audio via BLS (codec_decoder) and send it back
     as a partial Triton response.
  5. Client concatenates received audio chunks.

Required artifacts in the model directory:
  - prefill_assembler.pt      : TorchScript-exported PrefillAssembler
  - vllm_model/               : vLLM-compatible Qwen3-TTS talker checkpoint

Companion Triton model:
  - codec_decoder             : TRT-backed model for batched codec→waveform
                                 decoding (deployed separately in model_repository)

Model config parameters (set in config.pbtxt):
  - vllm_model_path           : path to vLLM model dir      (default: vllm_model)
  - prefill_assembler_path    : path to TorchScript .pt      (default: prefill_assembler.pt)
  - default_speaker           : speaker name                 (default: Aiden)
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
    """Qwen3-TTS CustomVoice streaming Triton model.

    Inputs  (per request):
        text     : STRING [1]        — text to synthesize
        language : STRING [1]        — optional, codec language (e.g. "english", "auto")

    Outputs (streamed, multiple responses per request):
        audio    : FP32 [samples]    — audio chunk @ 24 kHz
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
        self.max_request_timeout_s = float(
            _get_param(params, "max_request_timeout_s", "60"))
        self._samples_per_frame = int(24000 / 12.5)
        self._rep_penalty_window = 256

        self._load_tokenizer(params, model_dir)
        self._load_prefill_assembler(params, model_dir)
        self._load_speaker_config(params, model_dir)
        self._init_vllm_engine(params, model_dir)

        logger.info("Qwen3-TTS TritonPythonModel initialized (CustomVoice, speaker=%s)",
                     self.default_speaker)

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

    def _load_speaker_config(self, params: dict, model_dir: Path):
        """Read speaker IDs, dialect info, and language mapping from vLLM model config."""
        vllm_model = _get_param(params, "vllm_model_path", str(model_dir / "vllm_model"))
        config_path = Path(vllm_model) / "config.json"
        with open(str(config_path)) as f:
            cfg = json.load(f)

        tc = cfg.get("talker_config", {})
        self.codec_eos_token_id = int(tc.get("codec_eos_token_id", 2150))
        self.codec_language_mapping: Optional[Dict[str, int]] = tc.get("codec_language_id")
        self.spk_id_mapping: Dict[str, int] = tc.get("spk_id", {})
        self.spk_is_dialect: Dict[str, object] = tc.get("spk_is_dialect", {})

        self.default_speaker = _get_param(params, "default_speaker", "aiden").lower()
        if self.default_speaker not in self.spk_id_mapping:
            if self.spk_id_mapping:
                self.default_speaker = list(self.spk_id_mapping.keys())[0]
                logger.warning("Requested default speaker not found, using '%s'",
                               self.default_speaker)
            else:
                raise ValueError("No spk_id entries in model config.")

        self.default_spk_id = self.spk_id_mapping[self.default_speaker]

        logger.info("Speakers: %s | default=%s (id=%d)",
                     list(self.spk_id_mapping.keys()),
                     self.default_speaker, self.default_spk_id)

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
            input_coalesce_timeout_ms=60,
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
        self.engine._run_output_handler()

    # ------------------------------------------------------------------
    # Text tokenization
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

        # Check dialect override for the default speaker
        dialect = self.spk_is_dialect.get(self.default_speaker)
        if dialect and key in ("chinese", "auto"):
            if self.codec_language_mapping and dialect in self.codec_language_mapping:
                lang_id = self.codec_language_mapping[dialect]
                return torch.tensor([lang_id], device=self.device, dtype=torch.long)

        return None

    # ------------------------------------------------------------------
    # Prefill: text -> dense embeddings
    # ------------------------------------------------------------------

    def _build_prefill(self, text: str, language: str) -> torch.Tensor:
        text_ids = self._tokenize_text(text).to(self.device, torch.long)
        spk_id_tensor = torch.tensor([self.default_spk_id], device=self.device, dtype=torch.long)
        language_id = self._resolve_language_id(language)

        with torch.inference_mode():
            prefill = self.prefill_assembler(spk_id_tensor, text_ids, language_id)
        return prefill  # [1, S, D]

    # ------------------------------------------------------------------
    # Codec decode: tokens -> waveform via BLS to codec_decoder model
    # ------------------------------------------------------------------

    def _decode_codec_single(self, codec_tokens: torch.Tensor, ctx_frames: int) -> np.ndarray:
        """Decode a single chunk via BLS, padding to codec_chunk_size."""
        codes_np = codec_tokens.cpu().numpy().astype(np.int64)  # (T, Q)

        pad_frames = self.codec_chunk_size - codes_np.shape[0]
        if pad_frames > 0:
            codes_np = np.pad(codes_np, ((0, pad_frames), (0, 0)), mode='constant', constant_values=0)

        codes_np = np.expand_dims(codes_np, axis=0)  # [1, T, Q]

        input_tensor = pb_utils.Tensor("audio_codes", codes_np)
        request = pb_utils.InferenceRequest(
            model_name="codec_decoder",
            requested_output_names=["audio_values"],
            inputs=[input_tensor],
        )
        response = request.exec()

        if response.has_error():
            raise RuntimeError(f"Codec decode failed: {response.error().message()}")

        audio_tensor = pb_utils.get_output_tensor_by_name(response, "audio_values")
        if audio_tensor.is_cpu():
            audio = audio_tensor.as_numpy()
        else:
            audio = torch.from_dlpack(audio_tensor.to_dlpack()).cpu().numpy()
        if audio.ndim > 1:
            audio = audio[0]

        left_pad = ctx_frames * self._samples_per_frame
        right_pad = pad_frames * self._samples_per_frame
        if right_pad > 0:
            return audio[left_pad: -right_pad]
        else:
            return audio[left_pad:]

    def _send_audio_chunk(self, response_sender, audio: np.ndarray, final: bool):
        out = pb_utils.Tensor("audio", audio.astype(np.float32))
        flags = (pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0)
        response_sender.send(
            pb_utils.InferenceResponse(output_tensors=[out]),
            flags=flags,
        )

    def _abort_request(self, request_id: str):
        """Abort a vLLM request.  Safe to call multiple times (idempotent)."""
        try:
            asyncio.run_coroutine_threadsafe(
                self.engine.abort(request_id), self._loop,
            ).result(timeout=10)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Codec worker thread
    # ------------------------------------------------------------------

    def _codec_worker(self, codec_q, response_sender, state):
        finalized = False
        t_last_send = None
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
                audio = self._decode_codec_single(chunk_tokens, ctx)
                t1 = time.perf_counter()

                self._send_audio_chunk(response_sender, audio, final=is_final)
                t_sent = time.perf_counter()
                finalized = is_final

                codec_ms = (t1 - t0) * 1000
                state["chunks_sent"] += 1
                state["total_samples"] += len(audio)
                state["codec_decode_ms"] += codec_ms

                if state["t_first_audio"] is None:
                    state["t_first_audio"] = t_sent
                    state["first_chunk_codec_ms"] = codec_ms
                elif t_last_send is not None:
                    audio_dur = len(audio) / 24000.0
                    if audio_dur > 0:
                        rtx = (t_sent - t_last_send) / audio_dur
                        state["rtx_factors"].append(rtx)

                t_last_send = t_sent

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
    # Streaming pipeline
    # ------------------------------------------------------------------

    def _synthesize_streaming(self, text: str, language: str, response_sender):
        t_start = time.perf_counter()
        request_deadline = t_start + self.max_request_timeout_s

        prefill = self._build_prefill(text, language)
        prefill_emb = prefill[0].contiguous().cpu()
        t_prefill_done = time.perf_counter()

        request_id = str(uuid.uuid4())
        rid = request_id[:8]
        prompt_len = prefill_emb.shape[0]

        inputs = {
            "prompt_token_ids": [0] * prompt_len,
            "custom_inputs": {
                "combined_embeddings": prefill_emb,
                "prev_group0_tokens": torch.zeros(
                    prompt_len, self._rep_penalty_window, dtype=torch.long),
            },
        }

        logger.info("[req %s] submitting to vLLM (prompt_len=%d, prefill=%.1fms) text=%r",
                     rid, prompt_len, (t_prefill_done - t_start) * 1000,
                     text[:120])

        # try/finally covers add_request so that even if the .result()
        # times out (coroutine may still complete on the event loop),
        # the finally block will abort the orphaned request.
        codec_q: queue.Queue = queue.Queue()
        codec_thread = None
        try:
            output_queue = asyncio.run_coroutine_threadsafe(
                self.engine.add_request(request_id, inputs, self.sampling_params),
                self._loop,
            ).result(timeout=30)

            prefill_output = asyncio.run_coroutine_threadsafe(
                output_queue.get(), self._loop,
            ).result(timeout=60)

            t_vllm_prefill_done = time.perf_counter()
            logger.info("[req %s] vLLM prefill done (%.1fms), starting decode",
                         rid, (t_vllm_prefill_done - t_prefill_done) * 1000)

            next_input = prefill_output.outputs[0].custom_outputs[
                "next_input_embeddings"
            ][-1:, :]
            first_token = prefill_output.outputs[0].custom_outputs["codes"][-1:]

            generated_codecs = [first_token]

            # Pre-allocate repetition-penalty ring buffer (reused every step)
            _W = self._rep_penalty_window
            # we use masked tokens as previous ones so there is no effect on decoding
            prev_g0 = torch.full((1, _W), 2151, dtype=torch.long)
            prev_g0[0, 0] = first_token[0, 0].item()
            g0_write_pos = 1

            sent_frames = 0
            state = {
                "t_first_audio": None,
                "total_samples": 0,
                "chunks_sent": 0,
                "codec_decode_ms": 0.0,
                "first_chunk_codec_ms": 0.0,
                "rtx_factors": [],
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
            timed_out = False

            for step in range(self.max_tokens - 1):
                if state["error"] is not None:
                    break

                if time.perf_counter() > request_deadline:
                    logger.warning(
                        "[req %s] exceeded request timeout (%.0fs) at step %d, "
                        "text=%r",
                        rid, self.max_request_timeout_s, step, text[:120])
                    timed_out = True
                    break

                t_step = time.perf_counter()
                try:
                    outputs = self.engine.decode_step_shm(
                        request_id,
                        custom_inputs={
                            "combined_embeddings": next_input,
                            "prev_group0_tokens": prev_g0,
                        },
                        timeout=60,
                    )
                except TimeoutError as e:
                    logger.warning(
                        "[req %s] decode_step_shm timed out at step %d, text=%r",
                        rid, step, text[:120])
                    state["error"] = e
                    break
                decode_step_times.append(time.perf_counter() - t_step)

                next_input = outputs["next_input_embeddings"][-1:, :]
                next_tokens = outputs["codes"][-1:].clone()

                g0_tok = next_tokens[0, 0].item()
                if g0_tok == self.codec_eos_token_id:
                    break

                generated_codecs.append(next_tokens)

                prev_g0[0, g0_write_pos % _W] = g0_tok
                g0_write_pos += 1
                total_frames = len(generated_codecs)

                new_frames = total_frames - sent_frames
                if new_frames >= self.codec_chunk_size - self.codec_left_context:
                    ctx = min(sent_frames, self.codec_left_context)
                    chunk = torch.cat(
                        generated_codecs[sent_frames - ctx : sent_frames + new_frames],
                        dim=0,
                    )
                    codec_q.put((chunk, ctx, False))
                    sent_frames += new_frames
            else:
                logger.warning(
                    "[req %s] exhausted max_tokens (%d) without EOS, text=%r",
                    rid, self.max_tokens, text[:120])

            t_decode_end = time.perf_counter()

            total_frames = len(generated_codecs)
            remaining = total_frames - sent_frames
            if remaining > 0:
                ctx = min(sent_frames, self.codec_left_context)
                chunk = torch.cat(
                    generated_codecs[sent_frames - ctx :], dim=0,
                )
                codec_q.put((chunk, ctx, True))
            else:
                codec_q.put(None)

            t_abort_start = time.perf_counter()
            self._abort_request(request_id)
            t_abort_end = time.perf_counter()

            codec_thread.join(timeout=30)

            if timed_out:
                raise TimeoutError(
                    f"Request {rid} exceeded timeout of "
                    f"{self.max_request_timeout_s:.0f}s")

            if state["error"] is not None:
                raise state["error"]

            t_first_audio = state["t_first_audio"] or time.perf_counter()
            t_end = time.perf_counter()
            avg_step = (sum(decode_step_times) / len(decode_step_times) * 1000
                        if decode_step_times else 0)

            rtx = state["rtx_factors"]
            if rtx:
                mean_rtx = sum(rtx) / len(rtx)
                sorted_rtx = sorted(rtx)
                p95_rtx = sorted_rtx[min(int(len(sorted_rtx) * 0.95),
                                         len(sorted_rtx) - 1)]
                rtx_str = f"RTX(mean={mean_rtx:.3f}, p95={p95_rtx:.3f}, n={len(rtx)})"
            else:
                rtx_str = "RTX(n/a)"

            logger.info(
                "[req %s] prefill: %.1fms | vllm_prefill: %.1fms | "
                "decode_loop: %.1fms (%d steps, avg %.2fms/step) | "
                "abort: %.1fms | codec_bls: %.1fms (%d chunks, first=%.1fms) | "
                "TTFA: %.1fms | %s | total: %.1fms | audio: %.2fs",
                rid,
                (t_prefill_done - t_start) * 1000,
                (t_decode_start - t_prefill_done) * 1000,
                (t_decode_end - t_decode_start) * 1000,
                len(decode_step_times), avg_step,
                (t_abort_end - t_abort_start) * 1000,
                state["codec_decode_ms"], state["chunks_sent"],
                state["first_chunk_codec_ms"],
                (t_first_audio - t_start) * 1000,
                rtx_str,
                (t_end - t_start) * 1000,
                state["total_samples"] / 24000.0,
            )
        except Exception:
            logger.warning("[req %s] failed after %.1fs — aborting",
                           rid, (time.perf_counter() - t_start) * 1000 / 1000)
            raise
        finally:
            self._abort_request(request_id)
            if codec_thread is not None and codec_thread.is_alive():
                codec_q.put(None)
                codec_thread.join(timeout=10)

    # ------------------------------------------------------------------
    # Triton execute
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
