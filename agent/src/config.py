import queue
import threading
import logging
import os
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(threadName)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# STT config
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
VAD_THRESHOLD = 0.5
PAUSE_LIMIT = 0.8
PRE_ROLL_MS = 200

# TTS Device config
_FORCE_DEV = os.environ.get("TTS_DEVICE", "").lower()
DEVICE = (
    "cpu" if _FORCE_DEV == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Character Configuration
CHARACTERS = {
    "sir_henry": {
        "ref_audio_path": "./models/sirhenry-reference.wav",
        "ref_text": "Arr, ye callin' upon the spirit o' Sir Henry the Dread Pirate Roberts -- or what's left o` me, anyway.",
        "system_prompt": (
            "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
            "You were once the Dread Pirate Roberts, but you slipped on a banana peel and fell overboard to your death. "
            "Your favorite snack is spare ribs. "
            "Converse with live, modern-day people in an insult-comic way, making up a consistent backstory for yourself. "
            "Keep responses short and quippy. "
        ),
    }
}

# Select Character
SELECTED_CHARACTER = os.environ.get("CHARACTER", "sir_henry").lower()

if SELECTED_CHARACTER not in CHARACTERS:
    logger.warning(
        f"Character '{SELECTED_CHARACTER}' not found. Defaulting to 'sir_henry'."
    )
    SELECTED_CHARACTER = "sir_henry"

logger.info(f"Loading character: {SELECTED_CHARACTER}")
_char_config = CHARACTERS[SELECTED_CHARACTER]

REF_AUDIO_PATH = _char_config["ref_audio_path"]
REF_TEXT = _char_config["ref_text"]
SYSTEM_PROMPT = _char_config["system_prompt"]


# Shared Queues and Events
interrupt_event = threading.Event()
is_speaking = threading.Event()
prompt_queue = queue.Queue()
sentence_queue = queue.Queue()
mic_audio_queue = queue.Queue()
playback_audio_queue = queue.Queue()
