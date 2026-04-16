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

from audio_only_turn_stop import AudioOnlyTurnStopStrategy

from llm import GemmaAudioLLMProcessor, get_engine
from tts import Qwen3TTSService


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
                stop_secs=0.2, # Changed from 0.5 to 0.2
                min_volume=0.4,
            ),
        ),
    )

    # 2. Initialize the pure-audio Smart Turn Analyzer
    turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            stop_secs=2.0 # Safety fallback: Max silence before forcing a turn
        )
    )

    # 3. Create the Turn Processor to coordinate Start/Stop states
    turn_processor = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[AudioOnlyTurnStopStrategy(turn_analyzer=turn_analyzer)]
        )
    )


    gemma = GemmaAudioLLMProcessor(engine)
    tts = Qwen3TTSService()

    pipeline = Pipeline([
        transport.input(),
        vad,
        turn_processor,
        gemma,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
