"""Streaming ASR processor backed by NVIDIA Riva.

Audio is streamed to a Riva Speech Recognition gRPC endpoint as soon as the
VAD reports the start of a user turn (``VADUserStartedSpeakingFrame``).
Audio chunks (``InputAudioRawFrame``) are forwarded to Riva while the user
is speaking. When VAD reports end of speech (``VADUserStoppedSpeakingFrame``)
we close the audio side of the bidi stream so the server flushes its final
result.

Interim hypotheses are pushed downstream as ``InterimTranscriptionFrame``,
and the final hypothesis as ``TranscriptionFrame(finalized=True)`` — exactly
what ``TurnAnalyzerUserTurnStopStrategy`` needs to fire end-of-turn.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Optional

import riva.client
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


_END_OF_AUDIO = object()  # sentinel pushed into the audio queue to close the stream


class RivaStreamingASRProcessor(FrameProcessor):
    """Streaming Riva ASR, gated by the upstream VAD."""

    def __init__(
        self,
        *,
        server: str = "localhost:50051",
        language_code: str = "en-US",
        model_name: str = "",
        sample_rate: int = 16000,
        enable_automatic_punctuation: bool = True,
        max_alternatives: int = 1,
        user_id: str = "user",
        use_ssl: bool = False,
        ssl_root_cert: Optional[str] = None,
        ssl_client_cert: Optional[str] = None,
        ssl_client_key: Optional[str] = None,
    ):
        super().__init__()
        self._server = server
        self._language = language_code
        self._model = model_name
        self._sample_rate = sample_rate
        self._enable_automatic_punctuation = enable_automatic_punctuation
        self._max_alternatives = max_alternatives
        self._user_id = user_id

        self._auth = riva.client.Auth(
            ssl_root_cert=ssl_root_cert,
            ssl_client_cert=ssl_client_cert,
            ssl_client_key=ssl_client_key,
            use_ssl=use_ssl,
            uri=server,
        )
        self._asr_service = riva.client.ASRService(self._auth)

        # Per-stream state -- recreated on each user turn.
        self._audio_q: Optional[queue.Queue] = None  # threading queue, blocking
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_active: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_interim_text: str = ""

    # -- Pipecat lifecycle -------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._loop = asyncio.get_running_loop()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._close_stream()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._open_stream()

        elif isinstance(frame, InputAudioRawFrame):
            if self._stream_active and self._audio_q is not None:
                # Riva expects raw little-endian int16 PCM bytes.
                try:
                    self._audio_q.put_nowait(bytes(frame.audio))
                except queue.Full:
                    logger.warning("RivaASR: audio queue full, dropping chunk")

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._close_stream()

        await self.push_frame(frame, direction)

    # -- Stream management -------------------------------------------------

    def _build_streaming_config(self) -> riva.client.StreamingRecognitionConfig:
        return riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=self._sample_rate,
                language_code=self._language,
                model=self._model,
                max_alternatives=self._max_alternatives,
                enable_automatic_punctuation=self._enable_automatic_punctuation,
            ),
            interim_results=True,
        )

    def _open_stream(self) -> None:
        if self._stream_active:
            return

        self._audio_q = queue.Queue(maxsize=512)
        self._last_interim_text = ""
        self._stream_active = True

        config = self._build_streaming_config()
        loop = self._loop
        assert loop is not None, "RivaASR: event loop not captured (StartFrame missed)"

        self._stream_thread = threading.Thread(
            target=self._run_stream,
            args=(config, loop),
            daemon=True,
            name="RivaASRStream",
        )
        self._stream_thread.start()
        logger.debug("RivaASR: opened streaming session")

    def _close_stream(self) -> None:
        if not self._stream_active:
            return
        self._stream_active = False
        if self._audio_q is not None:
            try:
                self._audio_q.put_nowait(_END_OF_AUDIO)
            except queue.Full:
                # Best-effort: drain one item to make room and try again.
                try:
                    self._audio_q.get_nowait()
                    self._audio_q.put_nowait(_END_OF_AUDIO)
                except queue.Empty:
                    pass
        logger.debug("RivaASR: closing streaming session")

    # -- Background gRPC thread -------------------------------------------

    def _audio_chunks_iter(self):
        """Blocking generator pulled by the gRPC bidi stream."""
        assert self._audio_q is not None
        while True:
            chunk = self._audio_q.get()
            if chunk is _END_OF_AUDIO:
                return
            yield chunk

    def _run_stream(
        self,
        config: riva.client.StreamingRecognitionConfig,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        try:
            responses = self._asr_service.streaming_response_generator(
                audio_chunks=self._audio_chunks_iter(),
                streaming_config=config,
            )
            for response in responses:
                if not response.results:
                    continue
                for result in response.results:
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript
                    if not text:
                        continue
                    asyncio.run_coroutine_threadsafe(
                        self._emit_transcript(text, result.is_final), loop
                    )
        except Exception as e:  # pragma: no cover - defensive logging
            logger.exception(f"RivaASR: stream failed: {e}")

    async def _emit_transcript(self, text: str, is_final: bool) -> None:
        timestamp = _iso_now()
        if is_final:
            logger.info(f"RivaASR: final [{text}]")
            await self.push_frame(
                TranscriptionFrame(
                    text=text.strip(),
                    user_id=self._user_id,
                    timestamp=timestamp,
                    finalized=True,
                )
            )
            self._last_interim_text = ""
        else:
            if text == self._last_interim_text:
                return
            self._last_interim_text = text
            await self.push_frame(
                InterimTranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=timestamp,
                )
            )


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
