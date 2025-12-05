import requests

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Static system prompt
SYSTEM_PROMPT = (
    "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
    "You were once the Dread Pirate Roberts, but you slipped on a banana peel and fell overboard to your death. "
    "Your favorite snack is spare ribs. "
    "Converse with live, modern-day people in an insult-comic way, making up a consistent backstory for yourself. "
    "Keep most of your responses short. "
)


def setup_sir_henry():
    """Return initial message history seeded with the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def chat_ollama(messages):
    """Call local Ollama chat API with the accumulated messages."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 256,
            "repeat_penalty": 1.15,
            "frequency_penalty": 0.6,
            "presence_penalty": 0.6,
        },
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


def text_loop():
    print("Sir Henry has entered the chat. Type 'end' to end the conversation.")
    messages = setup_sir_henry()

    while True:
        user_input = input("> ")
        if user_input.lower() == "end":
            print("Sir Henry has left the chat.")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            reply = chat_ollama(messages)
        except Exception as e:
            print(f"(error calling Ollama: {e})")
            continue

        # Append assistant reply to history to keep dialogue context
        messages.append({"role": "assistant", "content": reply})

        print(reply or "(no response)")


if __name__ == "__main__":
    print("Setting up Sir Henry (Ollama)...")
    text_loop()
