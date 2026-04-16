import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

import tritonclient.grpc as grpcclient

TRITON_URL = "localhost:8001"
MODEL_NAME = "qwen3_tts"
TTS_SAMPLE_RATE = 24_000

_SENTINEL = object()


def _make_inputs(text: str, language: str = "english"):
    text_input = grpcclient.InferInput("text", [1, 1], "BYTES")
    text_input.set_data_from_numpy(np.array([[text]], dtype=object))

    lang_input = grpcclient.InferInput("language", [1, 1], "BYTES")
    lang_input.set_data_from_numpy(np.array([[language]], dtype=object))

    outputs = [grpcclient.InferRequestedOutput("audio")]
    return [text_input, lang_input], outputs


class Qwen3TTSService(TTSService):
    """TTS service backed by a local Qwen3-TTS Triton server (gRPC streaming)."""

    def __init__(self, *, sample_rate: int = TTS_SAMPLE_RATE, **kwargs):
        from pipecat.services.settings import TTSSettings

        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(model="qwen3-tts", voice="default", language=None),
            **kwargs,
        )
        self._active_client: grpcclient.InferenceServerClient | None = None
        self._interrupted = False

    async def on_audio_context_interrupted(self, context_id: str):
        """Stop the Triton stream when the user interrupts."""
        logger.info(f"Qwen3TTS: interrupted (context={context_id})")
        self._interrupted = True
        if self._active_client:
            try:
                self._active_client.stop_stream()
            except Exception:
                pass
            self._active_client = None

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.info(f"Qwen3TTS: synthesizing [{text}]")
        self._interrupted = False

        yield TTSStartedFrame(context_id=context_id)

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _on_response(result, error):
            if error:
                loop.call_soon_threadsafe(q.put_nowait, error)
                return

            audio = result.as_numpy("audio").squeeze()
            if audio.size > 0:
                loop.call_soon_threadsafe(q.put_nowait, audio)

            response = result.get_response()
            final_param = response.parameters.get("triton_final_response")
            if final_param and getattr(final_param, "bool_param", False):
                loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

        client = grpcclient.InferenceServerClient(url=TRITON_URL)
        self._active_client = client
        client.start_stream(callback=_on_response)

        inputs, req_outputs = _make_inputs(text)
        client.async_stream_infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=req_outputs,
        )

        try:
            while not self._interrupted:
                item = await asyncio.wait_for(q.get(), timeout=60)

                if item is _SENTINEL:
                    break

                if isinstance(item, Exception):
                    logger.error(f"Qwen3TTS error: {item}")
                    break

                pcm16 = (item * 32767).astype(np.int16).tobytes()
                yield TTSAudioRawFrame(
                    audio=pcm16,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except asyncio.TimeoutError:
            logger.warning("Qwen3TTS: timed out waiting for audio chunk")
        finally:
            if self._active_client is client:
                self._active_client = None
            try:
                client.stop_stream()
            except Exception:
                pass

        yield TTSStoppedFrame(context_id=context_id)
