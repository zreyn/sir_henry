from os import getenv
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from elevenlabs import ElevenLabs, stream

import speech_recognition as sr

load_dotenv()
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = getenv("ELEVENLABS_API_KEY")


elevenlabs_voice_id = "PPzYpIqttlTYA83688JI"
elevenlabs_model_id = "eleven_multilingual_v2"
model = OpenAIModel("gpt-4o")


def setup_sir_henry():
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
            "You were once the Dread Pirate Roberts, but you slipped on a banana peel and fell overboard to your death. "
            "Converse with live, modern-day people in a playful way, making up a consistent backstory for yourself. "
            "Keep your responses limited to a few sentences at a time please. "
        ),
    )

    listener = sr.Recognizer()
    tts_engine = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    return agent, tts_engine, listener


def text_loop(agent, tts_engine, listener):
    print("Sir Henry has entered the chat.")

    response = agent.run_sync("")
    print(response.data)
    text_to_speech(response.data, tts_engine)
    while True:
        try:
            user_input = speech_to_text(listener)
        except sr.UnknownValueError:
            text_to_speech("Sorry, I didn't catch that.", tts_engine)
        except Exception as e:
            print(f"An error occurred: {e}")

        response = agent.run_sync(user_input)
        print(response.data)
        text_to_speech(response.data, tts_engine)

    return


def text_to_speech(text, tts_engine):
    try:
        audio_stream = tts_engine.text_to_speech.convert_as_stream(
            voice_id=elevenlabs_voice_id,
            model_id=elevenlabs_model_id,
            text=text,
        )
        stream(audio_stream)
    except Exception as e:
        print(f"Error during TTS conversion: {e}")


def speech_to_text(listener):
    with sr.Microphone() as source:
        print("Listening...")
        listener.adjust_for_ambient_noise(source)
        voice = listener.listen(source)
        user_input = listener.recognize_whisper(voice)
        user_input = user_input.lower()
        print("Heard: " + user_input)
        return user_input


if __name__ == "__main__":
    print("Setting up Sir Henry...")
    agent, tts_engine, listener = setup_sir_henry()

    text_loop(agent, tts_engine, listener)
