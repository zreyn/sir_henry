from os import getenv
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

model = OpenAIModel("gpt-4o")


def setup_sir_henry(text_to_speech=False):
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
            "You were once the Dread Pirate Roberts, but you slipped on a banana peel and fell overboard to your death. "
            "Your favorite snack is spare ribs. "
            "Converse with live, modern-day people in an insult-comic way, making up a consistent backstory for yourself. "
            "Keep most of your responses short. "
        ),
    )

    return agent


def text_loop(agent):
    print("Sir Henry has entered the chat. Type 'end' to end the conversation.")
    response = agent.run_sync("")
    print(response.data)

    while True:
        user_input = input("> ").lower()

        if user_input == "end":
            print("Sir Henry has left the chat.")
            break

        response = agent.run_sync(user_input)
        print(response.data)

    return


if __name__ == "__main__":
    print("Setting up Sir Henry...")
    agent = setup_sir_henry()

    text_loop(agent)
