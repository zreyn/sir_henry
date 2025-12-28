import os
from dotenv import load_dotenv
import logging

from elevenlabs.client import ElevenLabs

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
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT")


def convert_tts_to_file(elevenlabs_client: ElevenLabs, text: str, file_name: str):

    logger.info(f"Converting {text}...")
    response = elevenlabs_client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        output_format=ELEVENLABS_OUTPUT_FORMAT,
        text=text,
        model_id=ELEVENLABS_MODEL_ID,
        # Optional voice settings that allow you to customize the output
        # voice_settings=VoiceSettings(
        #     stability=0.0,
        #     similarity_boost=1.0,
        #     style=0.0,
        #     use_speaker_boost=True,
        #     speed=1.0,
        # ),
    )

    with open(file_name, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    logger.info(f"... saved {file_name}")


def main():
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    phrase_output_dir = "raw/mp3"
    if not os.path.exists(phrase_output_dir):
        os.makedirs(phrase_output_dir, exist_ok=True)

    with open("raw/phrases.txt", "r") as f:
        phrases = f.readlines()

    logger.info(f"Read {len(phrases)} phrases...")

    for i, phrase in enumerate(phrases):
        file_name = f"{phrase_output_dir}/{i}.mp3"
        convert_tts_to_file(elevenlabs_client, phrase, file_name)


if __name__ == "__main__":
    main()
