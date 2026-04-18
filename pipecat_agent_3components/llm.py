import asyncio
import uuid

from loguru import logger
from transformers import AutoTokenizer

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
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

MODEL_PATH = "/home/vklimkov/workspace/vllm/models/Qwen3.5-4B"

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
                    max_model_len=1024,
                    gpu_memory_utilization=0.4,
                )
                _engine = AsyncLLM.from_engine_args(engine_args)
    return _engine


# ---------------------------------------------------------------------------
# Qwen3LLMProcessor
# ---------------------------------------------------------------------------


class Qwen3LLMProcessor(FrameProcessor):
    """Text-only LLM stage.

    Consumes finalized ``TranscriptionFrame``s produced by the ASR processor,
    accumulates them as the next user turn, runs Qwen3 via vLLM on
    ``UserStoppedSpeakingFrame``, and streams ``LLMTextFrame`` deltas to the
    TTS stage. History is bounded by ``max_turns`` to keep prompt length
    under control.
    """

    def __init__(self, engine: AsyncLLM, max_turns: int = 10):
        super().__init__()
        self._engine = engine
        self._sampling_params = SamplingParams(max_tokens=512, temperature=0.7)
        self._current_request_id: str | None = None

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self._messages: list[dict] = []
        self._max_turns = max_turns

        # Per-turn state.
        self._is_speaking = False
        self._pending_user_text: str = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(
            frame,
            (UserStartedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionFrame),
        ):
            logger.debug(
                f"Qwen3LLM: received {type(frame).__name__} dir={direction.name} "
                f"text={getattr(frame, 'text', None)!r} "
                f"finalized={getattr(frame, 'finalized', None)}"
            )

        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_speaking = True
            self._pending_user_text = ""
            await self._abort_current_request()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text and frame.finalized:
                self._pending_user_text = (
                    f"{self._pending_user_text} {text}".strip()
                    if self._pending_user_text
                    else text
                )
            await self.push_frame(frame, direction)

            # The TurnAnalyzerUserTurnStopStrategy only fires UserStoppedSpeakingFrame
            # *after* the final TranscriptionFrame arrives, so depending on frame
            # ordering through the controller we may receive the stop frame either
            # before or after this transcript. If we already saw the stop, kick
            # inference now.
            if (
                not self._is_speaking
                and frame.finalized
                and self._pending_user_text
                and self._current_request_id is None
            ):
                user_text = self._pending_user_text
                self._pending_user_text = ""
                await self._run_inference(user_text)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_speaking = False
            await self.push_frame(frame, direction)
            if self._pending_user_text:
                user_text = self._pending_user_text
                self._pending_user_text = ""
                await self._run_inference(user_text)
            else:
                logger.debug(
                    "Qwen3LLM: UserStoppedSpeakingFrame with no pending text yet "
                    "(waiting for finalized TranscriptionFrame)"
                )

        else:
            await self.push_frame(frame, direction)

    async def _abort_current_request(self):
        if self._current_request_id:
            rid = self._current_request_id
            self._current_request_id = None
            logger.info(f"Qwen3: aborting request {rid}")
            await self._engine.abort(rid)

    def _build_prompt(self) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._messages
        # Qwen3's chat template supports `enable_thinking=False` to suppress
        # the <think>...</think> reasoning block. Older transformers versions
        # don't accept the kwarg, so fall back gracefully.
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _reset_history(self):
        self._messages.clear()

    async def _send_context_reset_message(self):
        logger.warning("Context limit reached — resetting conversation history.")
        self._reset_history()
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(
            text="Our conversation got too long, so I've started fresh. "
                 "What can I help you with?"
        ))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _run_inference(self, user_text: str):
        self._messages.append({"role": "user", "content": user_text})

        user_turns = sum(1 for m in self._messages if m["role"] == "user")
        if user_turns > self._max_turns:
            self._messages.pop()
            await self._send_context_reset_message()
            return

        prompt = self._build_prompt()

        request_id = str(uuid.uuid4())
        self._current_request_id = request_id
        logger.info(f"Qwen3: user=[{user_text}] rid={request_id}")

        await self.push_frame(LLMFullResponseStartFrame())

        prev_text = ""
        full_text = ""

        try:
            async for output in self._engine.generate(
                {"prompt": prompt},
                sampling_params=self._sampling_params,
                request_id=request_id,
            ):
                if self._current_request_id != request_id:
                    logger.info(f"Qwen3: request {request_id} was interrupted")
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

        await self.push_frame(LLMFullResponseEndFrame())
