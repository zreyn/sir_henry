"""
Custom F5-TTS plugin for livekit-agents.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import AsyncIterator

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

from livekit.agents import tts, utils


@dataclass
class F5TTSOptions:
    ref_audio_path: str
    ref_text: str
    speed: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class F5TTS(tts.TTS):
    """
    F5-TTS plugin for LiveKit Agents.
    """

    def __init__(
        self,
        *,
        ref_audio_path: str,
        ref_text: str,
        speed: float = 1.0,
        device: str | None = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,  # F5-TTS outputs 24kHz audio
            num_channels=1,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._opts = F5TTSOptions(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            speed=speed,
            device=device,
        )

        self._model = None
        self._vocoder = None
        self._ref_audio = None
        self._ref_text = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        device = self._opts.device

        if torch.cuda.is_available() and device == "cuda":
            torch.zeros(1).cuda()

        # Download/load F5-TTS model
        model_dir = snapshot_download("SWivid/F5-TTS", cache_dir="./models/")
        base_dir = os.path.join(model_dir, "F5TTS_v1_Base")

        if os.path.exists(base_dir):
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
        else:
            raise FileNotFoundError(f"Model directory not found: {base_dir}")

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
            device=device,
        )

        # Load reference audio
        if os.path.exists(self._opts.ref_audio_path):
            self._ref_audio, self._ref_text = preprocess_ref_audio_text(
                self._opts.ref_audio_path, self._opts.ref_text
            )
        else:
            raise FileNotFoundError(
                f"Reference audio not found: {self._opts.ref_audio_path}"
            )

        self._loaded = True

    def synthesize(
        self,
        text: str,
        *,
        conn_options: tts.TTSConnOptions | None = None,
    ) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def _generate_audio(self, text: str) -> tuple[np.ndarray, int] | None:
        """Generate audio from text using F5-TTS."""
        self._ensure_loaded()

        if not text.strip():
            return None

        if self._ref_audio is None:
            return None

        try:
            audio, sample_rate, _ = infer_process(
                self._ref_audio,
                self._ref_text,
                text,
                self._model,
                self._vocoder,
                mel_spec_type="vocos",
                speed=self._opts.speed,
                device=self._opts.device,
            )
        except RuntimeError as e:
            msg = str(e)
            if "cudnn" in msg.lower():
                raise RuntimeError(
                    "TTS cuDNN error; set device=cpu to force CPU inference."
                ) from e
            raise

        # Normalize audio
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.squeeze(audio)

        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio, sample_rate


class ChunkedStream(tts.ChunkedStream):
    """Chunked stream implementation for F5-TTS."""

    def __init__(
        self,
        *,
        tts: F5TTS,
        input_text: str,
        conn_options: tts.TTSConnOptions | None = None,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self) -> None:
        """Generate audio and yield as a single chunk."""
        request_id = utils.shortuuid()

        # Run synthesis in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._tts._generate_audio, self._input_text
        )

        if result is None:
            return

        audio, sample_rate = result

        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        self._event_ch.send_nowait(
            tts.SynthesizedAudio(
                request_id=request_id,
                frame=utils.audio.AudioFrame(
                    data=audio_int16.tobytes(),
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio_int16),
                ),
            )
        )
