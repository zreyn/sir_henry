from os import getenv
from dotenv import load_dotenv
import nest_asyncio

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

nest_asyncio.apply()

model = OpenAIModel("gpt-4o")


def setup_sir_henry():
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
            "Converse with live, modern-day people in a playful way, making up a consistent backstory for yourself. "
            "Keep your responses limited to a few sentences at a time please. "
        ),
    )

    return agent


def text_loop(agent):
    print("Sir Henry has entered the chat. Type 'end' to end the conversation.")
    while True:
        user_input = input("> ").lower()

        if user_input == "end":
            print("Sir Henry has left the chat.")
            break

        response = agent.run_sync(user_input)
        print(response.data)

    return


def converse(agent):
    return


if __name__ == "__main__":
    print("Setting up Sir Henry...")
    agent = setup_sir_henry()

    text_loop(agent)
