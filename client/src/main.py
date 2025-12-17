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


async def _capture_microphone(audio_source: rtc.AudioSource, gain: float = 3.0) -> None:
    """
    Capture microphone audio and push it to the AudioSource.
    
    Args:
        audio_source: The LiveKit AudioSource to push frames to
        gain: Audio gain multiplier (default 3.0 = 3x amplification)
    """
    sample_rate = 48000
    channels = 1
    blocksize = 960  # 20ms frames at 48kHz
    
    loop = asyncio.get_event_loop()
    
    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        try:
            # Apply gain amplification and clip to prevent distortion
            amplified = np.clip(indata * gain, -1.0, 1.0)
            
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
    local_audio_track = rtc.LocalAudioTrack.create_audio_track("mic_track", audio_source)
    
    publish_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE,
    )
    await room.local_participant.publish_track(local_audio_track, publish_options)
    logger.info("Microphone track published.")

    # Start capturing microphone
    asyncio.create_task(_capture_microphone(audio_source, gain=40.0))
    logger.info("Microphone capture started.")

    # Keep alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await audio_source.aclose()
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
