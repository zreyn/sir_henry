import os
from dotenv import load_dotenv
import logging
import shutil

from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

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

def convert_mp3_to_wav(mp3_file_name: str, wav_file_name: str):
    logger.info(f"Converting {mp3_file_name} to {wav_file_name}...")
    try:
        sound = AudioSegment.from_mp3(f"{src_folder}/{file}")
        sound.export(output_path, format="wav")
    except Exception as e:
        logger.error(f"Failed to convert {src_folder}/{file}: {e}")
    logger.info(f"done.")


def main():
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    dataset_name = "sir_henry"

    mp3_output_dir = f"{dataset_name}/mp3"
    sir_henry/mp3"
    if not os.path.exists(mp3_output_dir):
        os.makedirs(mp3_output_dir, exist_ok=True)

    wav_output_dir = f"{dataset_name}/wavs"
    if not os.path.exists(wav_output_dir):
        os.makedirs(wav_output_dir, exist_ok=True)

    with open(f"{dataset_name}/phrases.txt", "r") as f:
        phrases = f.readlines()

    logger.info(f"Read {len(phrases)} phrases...")

    with open("sir_henry/metadata.csv", "w") as f:
        for i, phrase in enumerate(phrases):
            mp3_file_name = f"{mp3_output_dir}/{i}.mp3"
            convert_tts_to_file(elevenlabs_client, phrase, file_name)

            wav_file_name = f"{wav_output_dir}/{i}.wav"
            convert_mp3_to_wav(mp3_file_name, wav_file_name)

            f.write(f"wavs/{i}.wav|{phrase}")

    logger.info("Deleting mp3 files...")
    shutil.rmtree(mp3_output_dir)

    logger.info("Dataset ready!")


if __name__ == "__main__":
    main()
