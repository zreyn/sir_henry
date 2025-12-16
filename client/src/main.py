import asyncio
import logging
import os

from livekit import rtc
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

URL = os.environ.get("LIVEKIT_URL")
TOKEN = os.environ.get("LIVEKIT_TOKEN")


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
        async for frame in audio_stream:
            data = getattr(frame, "data", None)
            if data is None:
                continue
            stream.write(data)
    finally:
        stream.stop()
        stream.close()


async def main():
    room = rtc.Room()

    # 1. Handle Incoming Audio (Speakers)
    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"Subscribed to {participant.identity}'s audio")
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(_play_audio_stream(audio_stream))

    # 2. Connect
    print(f"Connecting to {URL}...")
    await room.connect(URL, TOKEN)
    print("Connected!")

    # 3. Publish Microphone
    local_audio_track = rtc.LocalAudioTrack.create_microphone_track("mic_track")
    await room.local_participant.publish_track(local_audio_track)
    print("Microphone published.")

    # Keep alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await room.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
