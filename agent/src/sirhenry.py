from os import getenv
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

import pyttsx3

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

model = OpenAIModel("gpt-4o")


def setup_sir_henry(text_to_speech=False):
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
            "Converse with live, modern-day people in a playful way, making up a consistent backstory for yourself. "
            "Keep your responses limited to a few sentences at a time please. "
        ),
    )

    tts_engine = None
    if text_to_speech:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 150)
        tts_engine.setProperty("volume", 1.0)
        # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Bahh") # shaky, gollum voice
        # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Whisper") # creepy whisper
        # tts_engine.setProperty("voice", "com.apple.eloquence.en-US.Rocko") # generic war games voice
        # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Albert") # creepy, strained voice
        # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Boing") # robot goblin
        # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Bubbles") # sounds underwater
        # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Cellos") # ominous cello singing
        tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Deranged") # shaky, crazy voice

    return agent, tts_engine


def text_loop(agent, tts_engine=None):
    print("Sir Henry has entered the chat. Type 'end' to end the conversation.")
    while True:
        user_input = input("> ").lower()

        if user_input == "end":
            print("Sir Henry has left the chat.")
            break

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


if __name__ == "__main__":
    print("Setting up Sir Henry...")
    agent, tts_engine = setup_sir_henry()

    text_loop(agent, tts_engine=tts_engine)
