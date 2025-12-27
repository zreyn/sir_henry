import os
from dotenv import load_dotenv
import logging

from elevenlabs import ElevenLabs, stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("elevenlabs-client")

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID")


def main():
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    audio_stream = elevenlabs_client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_MODEL_ID,
        text=sample_text,
    )
    stream(audio_stream)


if __name__ == "__main__":
    main()
