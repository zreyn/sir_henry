"""
LiveKit voice client for Sir Henry.

Connects to a LiveKit room, captures microphone audio, and plays back audio
from the voice agent.
"""

import asyncio
import logging
import os

from livekit import rtc
from livekit.api import (
    AccessToken,
    RoomAgentDispatch,
    RoomConfiguration,
    VideoGrants,
)
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

URL = os.environ.get("LIVEKIT_URL")
# If LIVEKIT_TOKEN is not provided, we can mint one locally using API key/secret
LIVEKIT_TOKEN = os.environ.get("LIVEKIT_TOKEN")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
LIVEKIT_ROOM = os.environ.get("LIVEKIT_ROOM", "testing")
AGENT_NAME = os.environ.get("AGENT_NAME", "voice-agent")
CLIENT_IDENTITY = os.environ.get("CLIENT_IDENTITY", "human-user")

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
    token = LIVEKIT_TOKEN
    if not token:
        if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
            raise RuntimeError(
                "LIVEKIT_TOKEN is not set and LIVEKIT_API_KEY/SECRET are missing. "
                "Set LIVEKIT_TOKEN, or provide LIVEKIT_API_KEY and LIVEKIT_API_SECRET to mint a token."
            )
        token = (
            AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(CLIENT_IDENTITY)
            .with_grants(VideoGrants(room_join=True, room=LIVEKIT_ROOM))
            .with_room_config(
                RoomConfiguration(
                    agents=[
                        RoomAgentDispatch(
                            agent_name=AGENT_NAME,
                            metadata='{"client_identity": "%s"}' % CLIENT_IDENTITY,
                        )
                    ]
                )
            )
            .to_jwt()
        )
        logger.info(
            f"Issued token for room '{LIVEKIT_ROOM}' with agent dispatch '{AGENT_NAME}'."
        )

    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to {participant.identity}'s audio")
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(_play_audio_stream(audio_stream))

    # Connect to LiveKit room
    logger.info(f"Connecting to {URL}...")
    await room.connect(URL, token)
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
