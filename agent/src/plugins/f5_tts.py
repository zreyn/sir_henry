"""
F5-TTS Plugin for LiveKit Agents
=================================

Local text-to-speech using F5-TTS with GPU acceleration.
No cloud APIs required - runs entirely on your hardware.

Features:
    - High-quality voice cloning from reference audio
    - GPU-accelerated synthesis (CUDA)
    - Configurable speed
    - DEBUG-level latency logging for benchmarking

Requirements:
    - f5-tts
    - Reference audio file (.wav) with transcript

Example:
    >>> from plugins import F5TTS
    >>>
    >>> tts = F5TTS(
    ...     ref_audio_path="/path/to/reference.wav",
    ...     ref_text="The transcript of the reference audio.",
    ...     speed=1.0
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import torch
from huggingface_hub import snapshot_download
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

from livekit.agents import tts, APIConnectOptions

if TYPE_CHECKING:
    from livekit.agents.tts.tts import AudioEmitter

__all__ = ["F5TTS"]

logger = logging.getLogger(__name__)


class _F5ChunkedStream(tts.ChunkedStream):
    """
    Internal ChunkedStream implementation for F5-TTS.

    Handles the async bridge between LiveKit's streaming interface
    and F5-TTS's synchronous synthesis.
    """

    def __init__(
        self,
        *,
        tts_plugin: F5TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=tts_plugin, input_text=input_text, conn_options=conn_options
        )
        self._f5_tts = tts_plugin

    async def _run(self, emitter: AudioEmitter) -> None:
        """Synthesize audio and emit it to LiveKit."""
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._f5_tts.sample_rate,
            num_channels=self._f5_tts.num_channels,
            mime_type="audio/pcm",
        )

        # Run blocking synthesis in thread pool
        start_time = time.perf_counter()
        loop = asyncio.get_running_loop()
        audio_bytes = await loop.run_in_executor(
            None, self._synthesize_blocking, self._input_text
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"TTS latency: {elapsed_ms:.0f}ms for {len(self._input_text)} chars"
        )

        if audio_bytes:
            emitter.push(audio_bytes)

    def _synthesize_blocking(self, text: str) -> bytes | None:
        """
        Blocking synthesis operation.

        Runs in a thread pool to avoid blocking the async event loop.
        """
        if not text.strip():
            return None

        # Ensure model is loaded
        self._f5_tts._ensure_loaded()

        if self._f5_tts._ref_audio is None:
            logger.warning("Reference audio not loaded")
            return None

        try:
            audio, sample_rate, _ = infer_process(
                self._f5_tts._ref_audio,
                self._f5_tts._ref_text,
                text,
                self._f5_tts._model,
                self._f5_tts._vocoder,
                mel_spec_type="vocos",
                speed=self._f5_tts.speed,
                device=self._f5_tts._device,
            )
        except RuntimeError as e:
            msg = str(e)
            if "cudnn" in msg.lower():
                logger.error("TTS cuDNN error; consider setting device='cpu'")
            raise

        # Convert to numpy array and normalize
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.squeeze(audio)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Convert float32 [-1, 1] to int16 PCM bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()


class F5TTS(tts.TTS):
    """
    LiveKit TTS plugin using F5-TTS for local speech synthesis with voice cloning.

    This plugin integrates the F5-TTS engine with LiveKit's Agents
    framework, enabling fully local text-to-speech with voice cloning
    from a reference audio sample.

    Args:
        ref_audio_path: Path to the reference audio file (.wav)
        ref_text: Transcript of the reference audio
        speed: Speech rate multiplier. 1.0 = normal, 1.5 = faster, 0.8 = slower
        device: Device for inference ("cuda" or "cpu"). Auto-detected if None.

    Example:
        >>> from plugins import F5TTS
        >>> from livekit.agents import AgentSession
        >>>
        >>> tts = F5TTS(
        ...     ref_audio_path="/models/reference.wav",
        ...     ref_text="Hello, this is my reference audio.",
        ...     speed=1.0
        ... )
        >>>
        >>> session = AgentSession(stt=..., llm=..., tts=tts)
    """

    def __init__(
        self,
        ref_audio_path: str,
        ref_text: str,
        speed: float = 1.0,
        device: str | None = None,
    ) -> None:
        # F5-TTS outputs 24kHz mono audio
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )

        self.speed = speed
        self._ref_audio_path = ref_audio_path
        self._ref_text_input = ref_text

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Lazy-loaded components
        self._model = None
        self._vocoder = None
        self._ref_audio = None
        self._ref_text = None
        self._loaded = False

        logger.info(f"F5-TTS initialized (device: {device}, speed: {speed})")

    def _ensure_loaded(self) -> None:
        """Lazy load the model on first use."""
        if self._loaded:
            return

        logger.info("Loading F5-TTS model...")

        if torch.cuda.is_available() and self._device == "cuda":
            torch.zeros(1).cuda()

        # Download/load F5-TTS model
        model_dir = snapshot_download("SWivid/F5-TTS", cache_dir="./models/")
        base_dir = os.path.join(model_dir, "F5TTS_v1_Base")

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Model directory not found: {base_dir}")

        safetensors_path = os.path.join(base_dir, "model_1250000.safetensors")
        pt_path = os.path.join(base_dir, "model_1250000.pt")
        vocab_path = os.path.join(base_dir, "vocab.txt")

        if os.path.exists(safetensors_path):
            ckpt_path = safetensors_path
        elif os.path.exists(pt_path):
            ckpt_path = pt_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {base_dir}")

        vocab_file = vocab_path if os.path.exists(vocab_path) else None

        self._vocoder = load_vocoder(
            is_local=True, local_path="./models/models--charactr--vocos-mel-24khz"
        )
        self._model = load_model(
            model_cls=DiT,
            model_cfg=dict(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
            ),
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self._device,
        )

        # Load reference audio
        if os.path.exists(self._ref_audio_path):
            self._ref_audio, self._ref_text = preprocess_ref_audio_text(
                self._ref_audio_path, self._ref_text_input
            )
            logger.info(f"Loaded reference audio: {self._ref_audio_path}")
        else:
            raise FileNotFoundError(
                f"Reference audio not found: {self._ref_audio_path}"
            )

        self._loaded = True
        logger.info("F5-TTS model loaded successfully")

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ) -> tts.ChunkedStream:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            conn_options: LiveKit connection options (optional)

        Returns:
            ChunkedStream that yields audio data
        """
        if conn_options is None:
            conn_options = APIConnectOptions()

        logger.debug(f"Synthesizing ({len(text)} chars): {text[:50]}...")

        return _F5ChunkedStream(
            tts_plugin=self,
            input_text=text,
            conn_options=conn_options,
        )
