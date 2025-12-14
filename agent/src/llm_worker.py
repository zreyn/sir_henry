import requests
import json
import re
import os
from config import (
    prompt_queue,
    sentence_queue,
    interrupt_event,
    SYSTEM_PROMPT,
    logger,
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost:11434")


def llm_worker():
    while True:
        user_text = prompt_queue.get()
        interrupt_event.clear()

        try:
            resp = requests.post(
                f"http://{OLLAMA_HOST}/api/generate",
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
                if "[SEP]" in buffer:
                    parts = buffer.split("[SEP]")
                    for part in parts[:-1]:
                        if part.strip():
                            sentence_queue.put(part.strip())
                            logger.info(
                                f"Queued sentence. Queue size: {sentence_queue.qsize()}"
                            )
                    buffer = parts[-1]
        except Exception as e:
            logger.error(f"LLM Error: {e}")
