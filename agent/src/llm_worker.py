import requests
import json
import re
from config import (
    prompt_queue,
    sentence_queue,
    interrupt_event,
    SYSTEM_PROMPT,
    logger,
)


def llm_worker():
    while True:
        user_text = prompt_queue.get()
        interrupt_event.clear()

        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": user_text,
                    "system": SYSTEM_PROMPT,
                    "stream": True,
                },
                stream=True,
            )
            buffer = ""
            for line in resp.iter_lines():
                if interrupt_event.is_set():
                    break
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                buffer += token
                if re.search(r"[\\.\\!\\?\\n]$", token):
                    sentence_queue.put(buffer.strip())
                    buffer = ""
        except Exception as e:
            logger.error(f"LLM Error: {e}")
