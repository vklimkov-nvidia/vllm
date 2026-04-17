"""Turn stop strategy for audio-only pipelines (no STT service).

The built-in TurnAnalyzerUserTurnStopStrategy waits for TranscriptionFrames
before firing user-turn-stopped. In an audio-to-audio pipeline there is no
STT, so that gate never opens and the system falls back to the 5-second
safety timeout.  This subclass bypasses the transcription requirement by
injecting a dummy text value when the acoustic model says the turn is done.
"""

from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


class AudioOnlyTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):

    async def _maybe_trigger_user_turn_stopped(self):
        if self._turn_complete:
            self._text = self._text or "[audio]"
            self._transcript_finalized = True
        await super()._maybe_trigger_user_turn_stopped()
