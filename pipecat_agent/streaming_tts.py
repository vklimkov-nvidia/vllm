import asyncio
import threading
import uuid
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService

import tritonclient.grpc as grpcclient

TRITON_URL = "localhost:8001"
MODEL_NAME = "qwen3_tts"
TTS_SAMPLE_RATE = 24_000


class _StreamSession:
    """One Triton bidi gRPC stream bound to a single TTS turn.

    Non-final text chunks are sent as they arrive; the server buffers them
    under ``stream_id`` and acks each chunk with an empty audio response.
    The final chunk (``end_of_text=True``) triggers synthesis, and audio
    chunks are streamed back on the same bidi stream.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, language: str):
        self._loop = loop
        self._language = language
        self._stream_id = uuid.uuid4().hex
        self._queues: dict[str, asyncio.Queue] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._client = grpcclient.InferenceServerClient(url=TRITON_URL)
        self._client.start_stream(callback=self._on_response)

    def _on_response(self, result, error):
        rid = None
        if result is not None:
            try:
                rid = result.get_response().id
                if isinstance(rid, bytes):
                    rid = rid.decode("utf-8")
            except Exception:
                rid = None

        if rid is None:
            if error:
                logger.error(f"Qwen3TTS stream error (no rid): {error}")
            return

        with self._lock:
            q = self._queues.get(rid)
        if q is None:
            return

        if error:
            self._loop.call_soon_threadsafe(q.put_nowait, ("error", error, True))
            return

        audio = None
        try:
            audio = result.as_numpy("audio")
            if audio is not None:
                audio = audio.squeeze()
        except Exception:
            audio = None

        resp = result.get_response()
        final_param = resp.parameters.get("triton_final_response")
        is_final = bool(final_param and getattr(final_param, "bool_param", False))
        self._loop.call_soon_threadsafe(q.put_nowait, ("audio", audio, is_final))

    def _send(self, text: str, end_of_text: bool, request_id: str):
        text_in = grpcclient.InferInput("text", [1, 1], "BYTES")
        text_in.set_data_from_numpy(np.array([[text]], dtype=object))
        lang_in = grpcclient.InferInput("language", [1, 1], "BYTES")
        lang_in.set_data_from_numpy(np.array([[self._language]], dtype=object))
        stream_in = grpcclient.InferInput("streaming", [1, 1], "BOOL")
        stream_in.set_data_from_numpy(np.array([[True]], dtype=bool))
        eot_in = grpcclient.InferInput("end_of_text", [1, 1], "BOOL")
        eot_in.set_data_from_numpy(np.array([[end_of_text]], dtype=bool))
        sid_in = grpcclient.InferInput("stream_id", [1, 1], "BYTES")
        sid_in.set_data_from_numpy(np.array([[self._stream_id]], dtype=object))

        self._client.async_stream_infer(
            model_name=MODEL_NAME,
            inputs=[text_in, lang_in, stream_in, eot_in, sid_in],
            outputs=[grpcclient.InferRequestedOutput("audio")],
            request_id=request_id,
        )

    def _register_request(self, request_id: str) -> asyncio.Queue:
        rq: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._queues[request_id] = rq
        return rq

    def _unregister_request(self, request_id: str):
        with self._lock:
            self._queues.pop(request_id, None)

    async def send_chunk_ack(self, text: str):
        """Send a non-final text chunk and wait for the server ack."""
        if self._closed or not text:
            return
        request_id = f"chunk-{uuid.uuid4().hex[:12]}"
        rq = self._register_request(request_id)
        try:
            self._send(text, end_of_text=False, request_id=request_id)
            while True:
                kind, payload, is_final = await rq.get()
                if kind == "error":
                    raise RuntimeError(f"Qwen3TTS server error: {payload}")
                if is_final:
                    return
        finally:
            self._unregister_request(request_id)

    async def stream_final(self) -> AsyncGenerator[np.ndarray, None]:
        """Send ``end_of_text=True`` and yield audio numpy arrays as they arrive."""
        if self._closed:
            return
        request_id = f"final-{uuid.uuid4().hex[:12]}"
        rq = self._register_request(request_id)
        try:
            self._send("", end_of_text=True, request_id=request_id)
            while True:
                kind, payload, is_final = await rq.get()
                if kind == "error":
                    raise RuntimeError(f"Qwen3TTS server error: {payload}")
                if payload is not None and payload.size > 0:
                    yield payload
                if is_final:
                    return
        finally:
            self._unregister_request(request_id)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._client.stop_stream()
        except Exception:
            pass


class Qwen3TTSService(TTSService):
    """TTS service backed by a local Qwen3-TTS Triton server with incremental text streaming.

    Uses TOKEN text aggregation so LLM deltas are forwarded to the TTS server
    as they arrive — no sentence accumulation. The server buffers the text
    under a turn-scoped ``stream_id`` and starts synthesis on ``end_of_text``,
    streaming audio back on the same bidi gRPC stream.
    """

    def __init__(
        self,
        *,
        sample_rate: int = TTS_SAMPLE_RATE,
        language: str = "english",
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            text_aggregation_mode=TextAggregationMode.TOKEN,
            push_start_frame=True,
            push_stop_frames=True,
            # Tokens can arrive slowly from the LLM; avoid spurious context
            # timeouts while we wait for the final flush.
            stop_frame_timeout_s=60.0,
            settings=TTSSettings(model="qwen3-tts", voice="default", language=None),
            **kwargs,
        )
        self._language = language
        self._sessions: dict[str, _StreamSession] = {}
        self._flush_tasks: dict[str, asyncio.Task] = {}

    def _get_or_open_session(self, context_id: str) -> _StreamSession:
        session = self._sessions.get(context_id)
        if session is None:
            session = _StreamSession(asyncio.get_running_loop(), self._language)
            self._sessions[context_id] = session
            logger.info(f"Qwen3TTS: opened stream for context {context_id}")
        return session

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        stripped = text.strip()
        if stripped:
            logger.debug(f"Qwen3TTS: chunk [{stripped!r}] ctx={context_id}")
            session = self._get_or_open_session(context_id)
            try:
                # Trailing space keeps word boundaries intact in the server-side buffer.
                await session.send_chunk_ack(stripped + " ")
            except Exception as e:
                logger.error(f"Qwen3TTS send_chunk failed: {e}")
                session.close()
                self._sessions.pop(context_id, None)
        return
        yield  # unreachable: makes this an async generator

    async def flush_audio(self, context_id: str | None = None):
        """Send end_of_text and stream audio back in a background task.

        Runs asynchronously so the caller (``on_turn_context_completed``)
        isn't blocked waiting for synthesis. The task feeds audio frames
        directly into the per-context audio queue and finalizes with a
        ``TTSStoppedFrame`` + ``remove_audio_context``.
        """
        if not context_id:
            return
        session = self._sessions.pop(context_id, None)
        if session is None:
            return
        existing = self._flush_tasks.pop(context_id, None)
        if existing and not existing.done():
            existing.cancel()
        task = asyncio.create_task(self._flush_audio_task(context_id, session))
        self._flush_tasks[context_id] = task

    async def _flush_audio_task(self, context_id: str, session: _StreamSession):
        logger.info(f"Qwen3TTS: finalizing context {context_id}")
        try:
            async for audio in session.stream_final():
                pcm16 = (audio * 32767).astype(np.int16).tobytes()
                await self.append_to_audio_context(
                    context_id,
                    TTSAudioRawFrame(
                        audio=pcm16,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    ),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Qwen3TTS: flush failed for ctx={context_id}: {e}")
        finally:
            try:
                if self.audio_context_available(context_id):
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
            except Exception:
                pass
            session.close()
            self._flush_tasks.pop(context_id, None)

    async def on_audio_context_interrupted(self, context_id: str):
        logger.info(f"Qwen3TTS: interrupted ctx={context_id}")
        task = self._flush_tasks.pop(context_id, None)
        if task and not task.done():
            task.cancel()
        session = self._sessions.pop(context_id, None)
        if session:
            session.close()
