from os import getenv
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

import pyttsx3

import speech_recognition as sr

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

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

    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 150)
    tts_engine.setProperty("volume", 1.0)
    tts_engine.setProperty("voice", "com.apple.voice.compact.en-GB.Daniel")

    listener = sr.Recognizer()

    return agent, tts_engine, listener


def text_loop(agent, tts_engine, listener):
    print("Sir Henry has entered the chat.")
    while True:
        try:
            user_input = speech_to_text(listener)
        except sr.UnknownValueError:
            text_to_speech("Sorry, I didn't catch that.", tts_engine)
        except Exception as e:
            print(f"An error occurred: {e}")

        response = agent.run_sync(user_input)
        print(response.data)
        if tts_engine:
            text_to_speech(response.data, tts_engine)

    return


def text_to_speech(text, tts_engine):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
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
