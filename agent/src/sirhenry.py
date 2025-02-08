
from os import getenv
from dotenv import load_dotenv

load_dotenv()
API_KEY = getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    print(f"OpenAI API Key: {API_KEY}")