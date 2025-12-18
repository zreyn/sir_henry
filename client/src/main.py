"""
LiveKit voice client for Sir Henry.

Connects to a LiveKit room, captures microphone audio, and plays back audio
from the voice agent.
"""

import asyncio
import logging
import os

from livekit import rtc
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

URL = os.environ.get("LIVEKIT_URL")
TOKEN = os.environ.get("LIVEKIT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sir_henry")


async def _play_audio_stream(audio_stream: rtc.AudioStream) -> None:
    """
    Consume incoming PCM frames from LiveKit and play them through the default
    system output using sounddevice.
    """
    stream = sd.RawOutputStream(
        samplerate=48000,
        channels=1,
        dtype="int16",
        blocksize=960,  # 20ms frames at 48kHz
    )
    stream.start()
    try:
        async for frame_event in audio_stream:
            audio_frame = frame_event.frame
            data = bytes(audio_frame.data)
            if data:
                stream.write(data)
    except Exception as e:
        logger.error(f"Error playing audio stream: {e}")
    finally:
        stream.stop()
        stream.close()


class AGC:
    """
    Automatic Gain Control - dynamically adjusts gain to maintain consistent output level.
    """

    def __init__(
        self,
        target_rms: float = 0.2,
        min_gain: float = 1.0,
        max_gain: float = 100.0,
        attack_time: float = 0.01,
        release_time: float = 0.1,
        noise_gate: float = 0.005,
    ):
        """
        Args:
            target_rms: Target RMS level (0.0-1.0, default 0.2 = -14 dBFS)
            min_gain: Minimum gain multiplier
            max_gain: Maximum gain multiplier (prevents amplifying pure noise)
            attack_time: How fast gain decreases when signal is loud (seconds)
            release_time: How fast gain increases when signal is quiet (seconds)
            noise_gate: RMS threshold below which signal is considered noise
        """
        self.target_rms = target_rms
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.attack_coef = attack_time
        self.release_coef = release_time
        self.noise_gate = noise_gate
        self.current_gain = max_gain / 2  # Start at moderate gain

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply AGC to audio frame."""
        # Calculate RMS of input
        rms = np.sqrt(np.mean(audio**2))

        # If below noise gate, don't adjust gain (just apply current)
        if rms > self.noise_gate:
            # Calculate desired gain to reach target RMS
            desired_gain = self.target_rms / (rms + 1e-10)
            desired_gain = np.clip(desired_gain, self.min_gain, self.max_gain)

            # Smooth gain changes (attack/release)
            if desired_gain < self.current_gain:
                # Signal is loud - decrease gain quickly (attack)
                self.current_gain += (
                    desired_gain - self.current_gain
                ) * self.attack_coef
            else:
                # Signal is quiet - increase gain slowly (release)
                self.current_gain += (
                    desired_gain - self.current_gain
                ) * self.release_coef

        # Apply gain and soft-clip to prevent harsh distortion
        amplified = audio * self.current_gain
        return np.tanh(amplified)  # Soft clipping via tanh


async def _capture_microphone(
    audio_source: rtc.AudioSource,
    target_level: float = 0.2,
    max_gain: float = 100.0,
) -> None:
    """
    Capture microphone audio with automatic gain control and push to AudioSource.

    Args:
        audio_source: The LiveKit AudioSource to push frames to
        target_level: Target RMS level for AGC (0.0-1.0, default 0.2)
        max_gain: Maximum gain multiplier for AGC (default 100.0)
    """
    sample_rate = 48000
    channels = 1
    blocksize = 960  # 20ms frames at 48kHz

    agc = AGC(target_rms=target_level, max_gain=max_gain)
    loop = asyncio.get_event_loop()

    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        try:
            # Apply automatic gain control
            amplified = agc.process(indata)

            # Convert float32 to int16
            audio_int16 = (amplified * 32767).astype("int16").flatten()

            frame = rtc.AudioFrame(
                data=audio_int16.tobytes(),
                sample_rate=sample_rate,
                num_channels=channels,
                samples_per_channel=frames,
            )
            asyncio.run_coroutine_threadsafe(audio_source.capture_frame(frame), loop)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    )
    stream.start()
    try:
        while True:
            await asyncio.sleep(0.1)
    finally:
        stream.stop()
        stream.close()


async def main():
    """Main entry point for the LiveKit client."""
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to {participant.identity}'s audio")
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(_play_audio_stream(audio_stream))

    # Connect to LiveKit room
    logger.info(f"Connecting to {URL}...")
    await room.connect(URL, TOKEN)
    logger.info("Connected!")

    # Create and publish microphone track
    audio_source = rtc.AudioSource(48000, 1)
    local_audio_track = rtc.LocalAudioTrack.create_audio_track(
        "mic_track", audio_source
    )

    publish_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE,
    )
    await room.local_participant.publish_track(local_audio_track, publish_options)
    logger.info("Microphone track published.")

    # Start capturing microphone with AGC
    asyncio.create_task(
        _capture_microphone(audio_source, target_level=0.25, max_gain=100.0)
    )
    logger.info("Microphone capture started with AGC.")

    # Keep alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await audio_source.aclose()
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
