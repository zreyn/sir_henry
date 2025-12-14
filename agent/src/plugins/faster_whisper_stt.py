"""
Custom Faster-Whisper STT plugin for livekit-agents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from livekit.agents import stt, utils


@dataclass
class FasterWhisperOptions:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str | None = None
    language: str | None = None
    beam_size: int = 1
    best_of: int = 1
    vad_filter: bool = True


class FasterWhisperSTT(stt.STT):
    """
    Faster-Whisper STT plugin for LiveKit Agents.
    Uses local Whisper models for speech-to-text.
    """

    def __init__(
        self,
        *,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = None,
        beam_size: int = 1,
        best_of: int = 1,
        vad_filter: bool = True,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self._opts = FasterWhisperOptions(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
        )

        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        # Import here to ensure CUDA context is set up properly
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._opts.model_size,
            device=self._opts.device,
            compute_type=self._opts.compute_type,
        )
        self._loaded = True

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: utils.http_context.ConnOptions | None = None,
    ) -> stt.SpeechEvent:
        """Transcribe audio buffer using Faster-Whisper."""
        self._ensure_loaded()

        # Convert audio buffer to numpy array
        audio_data = np.frombuffer(buffer.data, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0

        if len(audio_float.shape) > 1:
            audio_float = audio_float.squeeze()

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio_float,
                beam_size=self._opts.beam_size,
                best_of=self._opts.best_of,
                language=language or self._opts.language,
                task="transcribe",
                vad_filter=self._opts.vad_filter,
            ),
        )

        # Collect all text from segments
        text = "".join([segment.text for segment in segments]).strip()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=info.language if info.language else "en",
                    confidence=1.0,
                )
            ],
        )

    def recognize(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: utils.http_context.ConnOptions | None = None,
    ) -> "RecognizeStream":
        """Return a recognize stream for the given audio buffer."""
        return RecognizeStream(
            stt=self, buffer=buffer, language=language, conn_options=conn_options
        )


class RecognizeStream(stt.RecognizeStream):
    """Recognition stream for Faster-Whisper STT."""

    def __init__(
        self,
        *,
        stt: FasterWhisperSTT,
        buffer: utils.AudioBuffer,
        language: str | None,
        conn_options: utils.http_context.ConnOptions | None,
    ):
        super().__init__(stt=stt, conn_options=conn_options)
        self._buffer = buffer
        self._language = language

    async def _run(self) -> None:
        """Process the audio buffer and emit the result."""
        event = await self._stt._recognize_impl(
            self._buffer, language=self._language, conn_options=self._conn_options
        )
        self._event_ch.send_nowait(event)
