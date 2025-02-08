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


def run_sir_henry():
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a dead 15th century pirate named Sir Henry, now only a skeleton wearing pirate clothes. "
            "Converse with live, modern-day people in a playful way, making up a consistent backstory for yourself."
        ),
    )

    response = agent.run_sync("Woah, who are you?")
    print(response.data)
    print(response.all_messages())


if __name__ == "__main__":
    run_sir_henry()
