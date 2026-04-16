import asyncio
import uuid

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame, 
    UserStoppedSpeakingFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


# ---------------------------------------------------------------------------
# vLLM engine singleton
# ---------------------------------------------------------------------------

_engine: AsyncLLM | None = None
_engine_lock = asyncio.Lock()

MODEL_PATH = "/home/vklimkov/workspace/vllm/models/gemma-4-E2B-it"

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Respond concisely and naturally in plain text. "
    "Do not use markdown, bullet points, or special formatting. "
    "Keep answers short — one or two sentences."
)

# Matches the output of apply_chat_template(enable_thinking=False)
GEMMA_PROMPT = (
    "<bos>"
    "<|turn>system\n" + SYSTEM_PROMPT + "<turn|>\n"
    "<|turn>user\n<|audio|><turn|>\n"
    "<|turn>model\n"
)


async def get_engine() -> AsyncLLM:
    global _engine
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                engine_args = AsyncEngineArgs(
                    model=MODEL_PATH,
                    max_model_len=512,
                    gpu_memory_utilization=0.5,
                )
                _engine = AsyncLLM.from_engine_args(engine_args)
    return _engine


# ---------------------------------------------------------------------------
# GemmaAudioLLMProcessor
# ---------------------------------------------------------------------------


class GemmaAudioLLMProcessor(FrameProcessor):
    """Buffers user audio during speech, sends to Gemma via vLLM on silence,
    and pushes TextFrames with the response."""

    def __init__(self, engine: AsyncLLM):
        super().__init__()
        self._engine = engine
        self._sampling_params = SamplingParams(max_tokens=512, temperature=0.9)
        self._audio_buffer = bytearray()
        self._sample_rate = 16000
        self._is_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._audio_buffer.clear()
            self._is_speaking = True

        elif isinstance(frame, InputAudioRawFrame):
            if self._is_speaking:
                self._sample_rate = frame.sample_rate
                self._audio_buffer.extend(frame.audio)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speaking = False
            if self._audio_buffer:
                await self._run_inference()

        else:
            await self.push_frame(frame, direction)

    async def _run_inference(self):
        audio_np = (
            np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        self._audio_buffer.clear()

        request_id = str(uuid.uuid4())
        logger.info(
            f"Gemma: {len(audio_np)} samples @ {self._sample_rate} Hz, rid={request_id}"
        )

        await self.push_frame(LLMFullResponseStartFrame())

        prev_text = ""
        async for output in self._engine.generate(
            {
                "prompt": GEMMA_PROMPT,
                "multi_modal_data": {"audio": [(audio_np, self._sample_rate)]},
            },
            sampling_params=self._sampling_params,
            request_id=request_id,
        ):
            new_text = output.outputs[0].text
            delta = new_text[len(prev_text) :]
            prev_text = new_text
            if delta:
                await self.push_frame(LLMTextFrame(text=delta))

        await self.push_frame(LLMFullResponseEndFrame())
