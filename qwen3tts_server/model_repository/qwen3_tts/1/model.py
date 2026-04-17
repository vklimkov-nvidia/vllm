"""
Triton Python Model for Qwen3-TTS CustomVoice streaming inference.

Pipeline:
  1. Tokenize text into text_ids + group0_ids (batch prefill) or streaming prefill
     + per-step text_ids (HF-style incremental conditioning; see demo_qwen3_tts.ipynb).
  2. Run vLLM prefill + decode loop; stream codec chunks to codec_decoder via BLS
  3. Client concatenates received audio chunks @ 24 kHz

Multi-request text streaming (optional inputs ``streaming``, ``end_of_text``, ``stream_id``):
  When ``streaming`` is true **and** ``stream_id`` is non-empty, chunks sharing that id
  are buffered; when ``end_of_text`` is true the accumulated string is synthesized with
  incremental text conditioning (HF-style). Intermediate chunks get an empty final
  audio response (ack). If ``streaming`` is true but ``stream_id`` is missing or blank,
  the model falls back to the legacy path: one-shot synthesis of ``text`` with batch
  prefill (same as ``streaming`` false). Omitted optional inputs behave as before.
"""

import os
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from vllm.model_executor.models.qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

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

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        params = self.model_config.get("parameters", {})

        model_dir = Path(os.path.dirname(__file__))

        self.max_tokens = int(_get_param(params, "max_tokens", "2048"))
        self.codec_chunk_size = int(_get_param(params, "codec_chunk_size", "128"))
        self.codec_left_context = int(_get_param(params, "codec_left_context", "25"))
        self.first_chunk_frames = int(_get_param(params, "first_chunk_frames", "2"))
        self.max_request_timeout_s = float(_get_param(params, "max_request_timeout_s", "60"))
        self.text_stream_timeout_s = float(_get_param(params, "text_stream_timeout_s", "300"))
        self._samples_per_frame = int(24000 / 12.5)

        self._streaming_sessions: Dict[str, dict] = {}
        self._streaming_sessions_lock = threading.Lock()

        self._load_tokenizer(params, model_dir)
        self._load_speaker_config(params, model_dir)
        self._init_vllm_engine(params, model_dir)

        logger.info("Qwen3-TTS initialized (speaker=%s, speakers=%s)",
                     self.default_speaker, list(self.spk_id_mapping.keys()))

    def _load_tokenizer(self, params: dict, model_dir: Path):
        from transformers import AutoTokenizer
        vllm_model = _get_param(params, "vllm_model_path", str(model_dir / "vllm_model"))
        self.tokenizer = AutoTokenizer.from_pretrained(vllm_model, trust_remote_code=True)

    def _load_speaker_config(self, params: dict, model_dir: Path):
        vllm_model = _get_param(params, "vllm_model_path", str(model_dir / "vllm_model"))
        with open(str(Path(vllm_model) / "config.json")) as f:
            cfg = json.load(f)

        self._model_config_dict = cfg

        tc = cfg.get("talker_config", {})
        self.codec_eos_token_id = int(tc.get("codec_eos_token_id", 2150))
        self.tts_pad_token_id = int(cfg.get("tts_pad_token_id"))
        self.num_code_groups = int(tc.get("num_code_groups", 16))
        self.hidden_size = int(tc.get("hidden_size", 2048))
        self.codec_language_mapping: Optional[Dict[str, int]] = tc.get("codec_language_id")
        self.spk_id_mapping: Dict[str, int] = tc.get("spk_id", {})
        self.spk_is_dialect: Dict[str, object] = tc.get("spk_is_dialect", {})

        self.default_speaker = _get_param(params, "default_speaker", "aiden").lower()
        if self.default_speaker not in self.spk_id_mapping:
            if self.spk_id_mapping:
                self.default_speaker = list(self.spk_id_mapping.keys())[0]
                logger.warning("Requested speaker not found, using '%s'", self.default_speaker)
            else:
                raise ValueError("No spk_id entries in model config.")

    def _init_vllm_engine(self, params: dict, model_dir: Path):
        os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")

        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        vllm_model = _get_param(params, "vllm_model_path", str(model_dir / "vllm_model"))

        engine_args = AsyncEngineArgs(
            model=str(Path(vllm_model).absolute()),
            dtype=_get_param(params, "dtype", "bfloat16"),
            max_model_len=self.max_tokens,
            gpu_memory_utilization=float(_get_param(params, "gpu_memory_utilization", "0.6")),
            skip_tokenizer_init=True,
            enable_prefix_caching=False,
            trust_remote_code=True,
            input_coalesce_timeout_ms=30,
            #compilation_config={"cudagraph_mode": "PIECEWISE"},
            shm_decode=True,
            attention_backend="TRITON_ATTN",
        )

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self.engine = AsyncLLM.from_engine_args(engine_args)
        asyncio.run_coroutine_threadsafe(
            self._start_output_handler(), self._loop
        ).result(timeout=10)

        cfg = self._model_config_dict
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=cfg.get("temperature", 0.9),
            top_k=cfg.get("top_k", 50),
            top_p=cfg.get("top_p", 1.0),
            repetition_penalty=cfg.get("repetition_penalty", 1.0),
        )
        self._thread_pool = __import__("concurrent").futures.ThreadPoolExecutor(
            max_workers=int(_get_param(params, "max_concurrency", "8")),
        )

    async def _start_output_handler(self):
        self.engine._run_output_handler()

    def _resolve_language(self, language: str, speaker: str) -> Optional[str]:
        """Resolve language string for build_prefill_tokens."""
        key = language.strip().lower()
        if key == "auto":
            dialect = self.spk_is_dialect.get(speaker)
            if dialect and self.codec_language_mapping and dialect in self.codec_language_mapping:
                return dialect
            return None
        if self.codec_language_mapping and key in self.codec_language_mapping:
            return key
        return None

    def _decode_codec_single(self, codec_tokens: torch.Tensor, ctx_frames: int) -> np.ndarray:
        codes_np = codec_tokens.cpu().numpy().astype(np.int64)
        pad_frames = self.codec_chunk_size - codes_np.shape[0]
        if pad_frames > 0:
            codes_np = np.pad(codes_np, ((0, pad_frames), (0, 0)))

        response = pb_utils.InferenceRequest(
            model_name="codec_decoder",
            requested_output_names=["audio_values"],
            inputs=[pb_utils.Tensor("audio_codes", codes_np[np.newaxis])],
        ).exec()
        if response.has_error():
            raise RuntimeError(f"Codec decode failed: {response.error().message()}")

        audio_tensor = pb_utils.get_output_tensor_by_name(response, "audio_values")
        audio = (audio_tensor.as_numpy() if audio_tensor.is_cpu()
                 else torch.from_dlpack(audio_tensor.to_dlpack()).cpu().numpy())
        if audio.ndim > 1:
            audio = audio[0]

        left = ctx_frames * self._samples_per_frame
        right = pad_frames * self._samples_per_frame
        return audio[left:-right] if right > 0 else audio[left:]

    def _send_audio_chunk(self, response_sender, audio: np.ndarray, final: bool):
        response_sender.send(
            pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("audio", audio.astype(np.float32))]),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL if final else 0,
        )

    def _abort_request(self, request_id: str):
        try:
            asyncio.run_coroutine_threadsafe(
                self.engine.abort(request_id), self._loop,
            ).result(timeout=10)
        except Exception:
            pass

    def _codec_worker(self, codec_q, response_sender, state):
        finalized = False
        t_last_send = None
        try:
            while True:
                item = codec_q.get()
                if item is None:
                    self._send_audio_chunk(response_sender, np.array([], dtype=np.float32), final=True)
                    finalized = True
                    break
                chunk_tokens, ctx, is_final = item

                t0 = time.perf_counter()
                audio = self._decode_codec_single(chunk_tokens, ctx)
                codec_ms = (time.perf_counter() - t0) * 1000

                self._send_audio_chunk(response_sender, audio, final=is_final)
                t_sent = time.perf_counter()
                finalized = is_final

                state["chunks_sent"] += 1
                state["total_samples"] += len(audio)
                state["codec_ms"] += codec_ms

                if state["t_first_audio"] is None:
                    state["t_first_audio"] = t_sent
                elif t_last_send is not None:
                    audio_dur = len(audio) / 24000.0
                    if audio_dur > 0:
                        state["rtx_factors"].append((t_sent - t_last_send) / audio_dur)
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

    def _decode_text_id_for_step(
        self, step: int, incremental_text: bool, decode_schedule: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not incremental_text:
            return torch.tensor([self.tts_pad_token_id], dtype=torch.long)
        return Qwen3TTSTalkerForConditionalGeneration.streaming_decode_text_id_at_step(
            decode_schedule, step, self.tts_pad_token_id,
        )

    def _synthesize_streaming(
        self,
        text: str,
        language: str,
        speaker: str,
        response_sender,
        *,
        incremental_text: bool = False,
    ):
        t_start = time.perf_counter()
        request_deadline = t_start + self.max_request_timeout_s

        resolved_lang = self._resolve_language(language, speaker)
        decode_schedule: Optional[torch.Tensor] = None
        if incremental_text:
            text_ids, group0_ids = (
                Qwen3TTSTalkerForConditionalGeneration.build_streaming_prefill_tokens(
                    tokenizer=self.tokenizer,
                    text=text,
                    speaker=speaker,
                    language=resolved_lang,
                    config=self._model_config_dict,
                )
            )
            decode_schedule = (
                Qwen3TTSTalkerForConditionalGeneration.build_streaming_decode_text_ids(
                    tokenizer=self.tokenizer,
                    text=text,
                    config=self._model_config_dict,
                )
            )
        else:
            text_ids, group0_ids = Qwen3TTSTalkerForConditionalGeneration.build_prefill_tokens(
                tokenizer=self.tokenizer,
                text=text,
                speaker=speaker,
                language=resolved_lang,
                config=self._model_config_dict,
            )
        t_prefill = time.perf_counter()

        request_id = str(uuid.uuid4())
        rid = request_id[:8]
        prompt_len = text_ids.shape[0]

        codec_q: queue.Queue = queue.Queue()
        codec_thread = None
        state = {
            "t_first_audio": None, "total_samples": 0, "chunks_sent": 0,
            "codec_ms": 0.0, "rtx_factors": [], "error": None,
        }

        try:
            output_queue = asyncio.run_coroutine_threadsafe(
                self.engine.add_request(request_id, {
                    "prompt_token_ids": group0_ids.tolist(),
                    "custom_inputs": {
                        "text_ids": text_ids,
                        "prev_hidden": torch.zeros(
                            prompt_len, self.hidden_size, dtype=torch.bfloat16),
                    },
                }, self.sampling_params),
                self._loop,
            ).result(timeout=30)

            prefill_output = asyncio.run_coroutine_threadsafe(
                output_queue.get(), self._loop,
            ).result(timeout=60)
            t_vllm_prefill = time.perf_counter()

            custom_out = prefill_output.outputs[0].custom_outputs

            prev_hidden = custom_out["hidden"][-1:].clone()
            prev_group0 = prefill_output.outputs[0].token_ids[-1]
            sent_frames = 0

            codec_thread = threading.Thread(
                target=self._codec_worker, args=(codec_q, response_sender, state), daemon=True)
            codec_thread.start()

            generated_codecs = []
            decode_step_times = []
            timed_out = False

            for step in range(self.max_tokens - prompt_len - 1):
                if state["error"] is not None:
                    break
                if time.perf_counter() > request_deadline:
                    timed_out = True
                    break

                t_step = time.perf_counter()
                try:
                    decode_text_id = self._decode_text_id_for_step(
                        step, incremental_text, decode_schedule,
                    )
                    outputs = self.engine.decode_step_shm(
                        request_id,
                        custom_inputs={
                            "text_ids": decode_text_id,
                            "prev_hidden": prev_hidden,
                        },
                        timeout=60,
                    )
                except TimeoutError as e:
                    state["error"] = e
                    break
                decode_step_times.append(time.perf_counter() - t_step)

                codes_1_15 = outputs["codes"][-1:].clone()  # [1, 15]
                prev_hidden = outputs["hidden"][-1:].clone()  # [1, hidden_size]

                # Assemble complete codec frame: [group0, codes_1_15]
                frame = torch.cat([
                    torch.tensor([[prev_group0]], dtype=torch.long),
                    codes_1_15,
                ], dim=-1)  # [1, 16]
                generated_codecs.append(frame)

                sampled_token = outputs.get("sampled_token_ids")
                g0_tok = int(sampled_token[-1])

                if g0_tok == self.codec_eos_token_id:
                    break

                prev_group0 = g0_tok

                new_frames = len(generated_codecs) - sent_frames
                if sent_frames == 0:
                    threshold = self.first_chunk_frames
                else:
                    threshold = self.codec_chunk_size - self.codec_left_context
                if new_frames >= threshold:
                    ctx = min(sent_frames, self.codec_left_context)
                    codec_q.put((
                        torch.cat(generated_codecs[sent_frames - ctx:sent_frames + new_frames], dim=0),
                        ctx, False))
                    sent_frames += new_frames

            t_decode_end = time.perf_counter()

            remaining = len(generated_codecs) - sent_frames
            if remaining > 0:
                ctx = min(sent_frames, self.codec_left_context)
                codec_q.put((torch.cat(generated_codecs[sent_frames - ctx:], dim=0), ctx, True))
            else:
                codec_q.put(None)

            self._abort_request(request_id)
            t_abort = time.perf_counter()
            codec_thread.join(timeout=30)

            if timed_out:
                raise TimeoutError(f"Request {rid} exceeded {self.max_request_timeout_s:.0f}s timeout")
            if state["error"] is not None:
                raise state["error"]

            t_end = time.perf_counter()
            n_steps = len(decode_step_times)
            avg_step = (sum(decode_step_times) / n_steps * 1000) if n_steps else 0
            rtx = state["rtx_factors"]
            rtx_mean = sum(rtx) / len(rtx) if rtx else 0
            rtx_p95 = sorted(rtx)[min(int(len(rtx) * 0.95), len(rtx) - 1)] if rtx else 0

            logger.info(
                "Streaming rid=%s text_mode=%s tokenize=%.1fms vllm_prefill=%.1fms "
                "decode=%.1fms steps=%d step_avg=%.2fms "
                "abort=%.1fms codec=%.1fms chunks=%d "
                "ttfa=%.1fms rtx_mean=%.3f rtx_p95=%.3f "
                "total=%.1fms audio=%.2fs speaker=%s text=%r",
                rid,
                "incr" if incremental_text else "batch",
                (t_prefill - t_start) * 1000,
                (t_vllm_prefill - t_prefill) * 1000,
                (t_decode_end - t_vllm_prefill) * 1000,
                n_steps, avg_step,
                (t_abort - t_decode_end) * 1000,
                state["codec_ms"], state["chunks_sent"],
                ((state["t_first_audio"] or t_end) - t_start) * 1000,
                rtx_mean, rtx_p95,
                (t_end - t_start) * 1000,
                state["total_samples"] / 24000.0,
                speaker,
                text[:120],
            )
        except Exception:
            logger.error("Streaming rid=%s failed after %.1fms",
                         rid, (time.perf_counter() - t_start) * 1000)
            raise
        finally:
            self._abort_request(request_id)
            if codec_thread is not None and codec_thread.is_alive():
                codec_q.put(None)
                codec_thread.join(timeout=10)

    def _prune_stale_text_streams(self):
        now = time.time()
        with self._streaming_sessions_lock:
            stale = [
                sid for sid, s in self._streaming_sessions.items()
                if now - s["t_updated"] > self.text_stream_timeout_s
            ]
            for sid in stale:
                self._streaming_sessions.pop(sid, None)
        for sid in stale:
            logger.warning(
                "Dropped stale incremental-text session stream_id=%r (timeout=%.0fs)",
                sid, self.text_stream_timeout_s,
            )

    def _apply_text_stream_chunk(
        self,
        stream_id: str,
        text: str,
        end_of_text: bool,
        language: str,
        speaker: str,
    ) -> Optional[tuple[str, str, str]]:
        """Return None if this RPC only accumulates text (caller sends ack). Otherwise
        ``(full_text, language, speaker)`` for synthesis after ``end_of_text``."""
        self._prune_stale_text_streams()
        spk_norm = (speaker or self.default_speaker).lower()
        with self._streaming_sessions_lock:
            if stream_id not in self._streaming_sessions:
                self._streaming_sessions[stream_id] = {
                    "buffer": "",
                    "language": language,
                    "speaker": spk_norm,
                    "t_updated": time.time(),
                }
            sess = self._streaming_sessions[stream_id]
            sess["t_updated"] = time.time()
            sess["buffer"] += text
            if not end_of_text:
                return None
            full = sess["buffer"]
            lang_o = sess["language"]
            spk_o = sess["speaker"]
            del self._streaming_sessions[stream_id]
        return (full, lang_o, spk_o)

    def _handle_streaming_text_request(
        self,
        stream_id: str,
        text: str,
        end_of_text: bool,
        language: str,
        speaker: str,
        response_sender,
    ):
        try:
            merged = self._apply_text_stream_chunk(
                stream_id, text, end_of_text, language, speaker,
            )
            if merged is None:
                self._send_audio_chunk(
                    response_sender, np.array([], dtype=np.float32), final=True,
                )
                return
            full_text, lang, spk = merged
            if not full_text.strip():
                raise ValueError(
                    "Incremental text streaming: transcript is empty after end_of_text "
                    "(send non-empty text chunks before end_of_text).",
                )
            self._synthesize_streaming(
                full_text, lang, spk, response_sender, incremental_text=True,
            )
        except Exception as e:
            logger.error("Incremental text streaming failed: %s", e, exc_info=True)
            try:
                response_sender.send(
                    pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(e))),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
            except Exception:
                pass

    def execute(self, requests):
        for request in requests:
            response_sender = request.get_response_sender()
            try:
                text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy().flatten()[0].decode("utf-8")
                lang_tensor = pb_utils.get_input_tensor_by_name(request, "language")
                language = lang_tensor.as_numpy().flatten()[0].decode("utf-8") if lang_tensor else "auto"
                spk_tensor = pb_utils.get_input_tensor_by_name(request, "speaker")
                speaker = (spk_tensor.as_numpy().flatten()[0].decode("utf-8")
                           if spk_tensor else self.default_speaker)
                streaming_tensor = pb_utils.get_input_tensor_by_name(request, "streaming")
                streaming = bool(streaming_tensor.as_numpy().flatten()[0]) if streaming_tensor else False
                eot_tensor = pb_utils.get_input_tensor_by_name(request, "end_of_text")
                end_of_text = bool(eot_tensor.as_numpy().flatten()[0]) if eot_tensor else False
                sid_tensor = pb_utils.get_input_tensor_by_name(request, "stream_id")
                stream_id = (
                    sid_tensor.as_numpy().flatten()[0].decode("utf-8") if sid_tensor else ""
                )
                sid = stream_id.strip()
                use_chunked_text = streaming and bool(sid)
                if use_chunked_text:
                    self._thread_pool.submit(
                        self._handle_streaming_text_request,
                        sid, text, end_of_text, language, speaker, response_sender,
                    )
                else:
                    self._thread_pool.submit(self._handle_request, text, language, speaker, response_sender)
            except Exception as e:
                logger.error("Request parse failed: %s", e, exc_info=True)
                response_sender.send(
                    pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(e))),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
        return None

    def _handle_request(self, text: str, language: str, speaker: str, response_sender):
        try:
            self._synthesize_streaming(text, language, speaker, response_sender)
        except Exception as e:
            logger.error("Streaming failed: %s", e, exc_info=True)
            try:
                response_sender.send(
                    pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(e))),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
            except Exception:
                pass

    def finalize(self):
        if hasattr(self, "_loop") and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=10)
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown(wait=False)
        if hasattr(self, "_streaming_sessions_lock"):
            with self._streaming_sessions_lock:
                self._streaming_sessions.clear()
