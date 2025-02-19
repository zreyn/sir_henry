from os import getenv
from dotenv import load_dotenv

from elevenlabs import ElevenLabs, stream

load_dotenv()
ELEVENLABS_API_KEY = getenv("ELEVENLABS_API_KEY")


elevenlabs_voice_id = "PPzYpIqttlTYA83688JI"
elevenlabs_model_id = "eleven_multilingual_v2"

sample_text = (
    "Ahoy there! Seems the winds have carried me to your time! I'm Sir Henry, once known far and wide as "
    "the Dread Pirate Roberts. Alas, a traitorous banana peel took me down to Davey Jones's locker. Now I'm "
    "just a cheerful skeleton looking for a bit of company and perhaps a good pirate yarn or two. What tales "
    "can ye share with an old seadog like me?"
)


def setup_cool_pirate_voice():
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)


if __name__ == "__main__":
    print("Setting up cool pirate voice...")
    elevenlabs_client = setup_cool_pirate_voice()

    audio_stream = elevenlabs_client.text_to_speech.convert_as_stream(
        voice_id=elevenlabs_voice_id,
        model_id=elevenlabs_model_id,
        text=sample_text,
    )

    stream(audio_stream)
