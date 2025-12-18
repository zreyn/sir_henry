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

    # Create and publish microphone track using built-in WebRTC capture (AEC/NS/AGC)
    audio_source = rtc.AudioSource.capture()
    local_audio_track = rtc.LocalAudioTrack.create_audio_track("mic_track", audio_source)
    await room.local_participant.publish_track(local_audio_track)
    logger.info("Microphone track published (WebRTC capture).")

    # Keep alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
