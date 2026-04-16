import asyncio
import uuid

import numpy as np
from loguru import logger
from transformers import AutoTokenizer

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
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
    "Do not use markdown, bullet points, or special formatting."
)


async def get_engine() -> AsyncLLM:
    global _engine
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                engine_args = AsyncEngineArgs(
                    model=MODEL_PATH,
                    max_model_len=4096,
                    gpu_memory_utilization=0.5,
                )
                _engine = AsyncLLM.from_engine_args(engine_args)
    return _engine


# ---------------------------------------------------------------------------
# GemmaAudioLLMProcessor
# ---------------------------------------------------------------------------


class GemmaAudioLLMProcessor(FrameProcessor):
    """Buffers user audio, maintains multi-turn history, sends to Gemma via
    vLLM, and resets context when limits are reached."""

    def __init__(self, engine: AsyncLLM, max_turns: int = 5):
        super().__init__()
        self._engine = engine
        self._sampling_params = SamplingParams(max_tokens=512, temperature=0.9)
        self._audio_buffer = bytearray()
        self._sample_rate = 16000
        self._is_speaking = False
        self._current_request_id: str | None = None

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # HF chat-format messages; user audio turns use content=[{"type":"audio"}],
        # assistant turns use content="text".
        self._messages: list[dict] = []
        # Parallel list of (waveform, sample_rate) — one entry per user audio turn,
        # kept in the same order as `<|audio|>` placeholders appear in the prompt.
        self._audio_clips: list[tuple[np.ndarray, int]] = []
        self._max_turns = max_turns

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._audio_buffer.clear()
            self._is_speaking = True
            await self._abort_current_request()

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

    async def _abort_current_request(self):
        if self._current_request_id:
            rid = self._current_request_id
            self._current_request_id = None
            logger.info(f"Gemma: aborting request {rid}")
            await self._engine.abort(rid)

    def _build_prompt(self) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._messages
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _reset_history(self):
        self._messages.clear()
        self._audio_clips.clear()

    async def _send_context_reset_message(self):
        logger.warning("Context limit reached — resetting conversation history.")
        self._reset_history()
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(
            text="Our conversation got too long, so I've started fresh. "
                 "What can I help you with?"
        ))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _run_inference(self):
        audio_np = (
            np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        self._audio_buffer.clear()

        self._messages.append({
            "role": "user",
            "content": [{"type": "audio"}],
        })
        self._audio_clips.append((audio_np, self._sample_rate))

        user_turns = sum(1 for m in self._messages if m["role"] == "user")
        if user_turns > self._max_turns:
            await self._send_context_reset_message()
            return

        prompt = self._build_prompt()
        audio_list = list(self._audio_clips)

        request_id = str(uuid.uuid4())
        self._current_request_id = request_id
        logger.info(f"Gemma: {len(audio_list)} audio turn(s), rid={request_id}")

        await self.push_frame(LLMFullResponseStartFrame())

        prev_text = ""
        full_text = ""

        try:
            async for output in self._engine.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": audio_list},
                },
                sampling_params=self._sampling_params,
                request_id=request_id,
            ):
                if self._current_request_id != request_id:
                    logger.info(f"Gemma: request {request_id} was interrupted")
                    break

                new_text = output.outputs[0].text
                full_text = new_text
                delta = new_text[len(prev_text):]
                prev_text = new_text

                if delta:
                    await self.push_frame(LLMTextFrame(text=delta))

        except ValueError as e:
            err = str(e).lower()
            if any(k in err for k in ("context", "maximum", "length", "too long")):
                self._messages.pop()
                self._audio_clips.pop()
                await self._send_context_reset_message()
                return
            raise

        finally:
            if self._current_request_id == request_id:
                self._current_request_id = None

            if full_text:
                self._messages.append({"role": "assistant", "content": full_text})
            elif self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
                self._audio_clips.pop()

        await self.push_frame(LLMFullResponseEndFrame())
