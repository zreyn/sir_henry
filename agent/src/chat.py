import requests

# --- Configuration ---
# Ollama model name (must match what you have in Ollama)
OLLAMA_MODEL = "llama3.2:3b"  # Adjust this to match your Ollama model name
OLLAMA_BASE_URL = "http://localhost:11434"

# System prompt for Sir Henry
SYSTEM_PROMPT = (
    "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
    "You were once the Dread Pirate Roberts, but you slipped on a banana peel and fell overboard to your death. "
    "Your favorite snack is spare ribs. "
    "Converse with live, modern-day people in an insult-comic way, making up a consistent backstory for yourself. "
    "Keep most of your responses short. "
)


class LLMStreamer:
    def __init__(self, model_name=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        
        # Check if Ollama is running
        print(f"Connecting to Ollama at {base_url}...")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"Found {len(models)} model(s) in Ollama")
            
            # Check if our model is available
            if model_name not in model_names:
                print(f"Warning: Model '{model_name}' not found in Ollama.")
                print(f"Available models: {', '.join(model_names)}")
                if model_names:
                    print(f"Using first available model: {model_names[0]}")
                    self.model_name = model_names[0]
                else:
                    raise RuntimeError("No models found in Ollama. Please pull a model first.")
            else:
                print(f"Using model: {model_name}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {base_url}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except Exception as e:
            raise RuntimeError(f"Error connecting to Ollama: {e}")
        
        print("Ollama ready!")

    def generate(self, prompt):
        """
        Generate a complete response from the LLM with the Sir Henry system prompt.
        Returns the full response as a string.
        """
        # Ollama API format
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 256,  # max tokens
            },
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Extract the message content
            message = result.get("message", {})
            text = message.get("content", "").strip()
            
            return text
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama: {e}"


if __name__ == "__main__":
    llm = LLMStreamer()

    print(f"--- Connected to Ollama ({llm.model_name}) ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        print("\nSir Henry:")
        print("-" * 30)
        response = llm.generate(user_input)
        print(response)
        print("-" * 30)
        print("\n")
