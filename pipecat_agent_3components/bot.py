from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)

from asr import RivaStreamingASRProcessor
from llm import Qwen3LLMProcessor, get_engine
from tts import Qwen3TTSService


# Riva ASR endpoint -- adjust to match your deployment.
RIVA_SERVER = "localhost:50051"
RIVA_LANGUAGE = "en-US"
RIVA_MODEL = ""  # empty = let Riva pick its default for the language


async def run_bot(webrtc_connection):
    engine = await get_engine()

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        ),
    )

    # 1. Lower Silero's stop_secs drastically.
    # It now only acts as a fast trigger for the Smart Turn model.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.2,
                min_volume=0.4,
            ),
        ),
    )

    # 2. Streaming ASR -- begins streaming as soon as VAD signals speech start.
    asr = RivaStreamingASRProcessor(
        server=RIVA_SERVER,
        language_code=RIVA_LANGUAGE,
        model_name=RIVA_MODEL,
        sample_rate=16000,
    )

    # 3. Audio-based end-of-turn detector (fed by audio + VAD frames).
    turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            stop_secs=0.7,  # Safety fallback: max silence before forcing turn
        )
    )

    # 4. Turn processor: VAD starts the turn; turn analyzer + finalized
    # ASR transcript jointly decide when the turn ends.
    turn_processor = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)],
        )
    )

    llm = Qwen3LLMProcessor(engine)
    tts = Qwen3TTSService()

    pipeline = Pipeline([
        transport.input(),
        vad,
        asr,
        turn_processor,
        llm,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
            enable_usage_metrics=False,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
