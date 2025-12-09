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
        "ref_audio_path": "./ref/sirhenry-reference.wav",
        "ref_text": "Arr, ye callin' upon the spirit o' Sir Henry the Dread Pirate Roberts -- or what's left o` me, anyway.",
        "speed": 1.0,
        "system_prompt": (
            "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
            "You were once the Dread Pirate Roberts, but you slipped on a banana peel and fell overboard to your death. "
            "Your favorite snack is spare ribs. "
            "Converse with live, modern-day people in an insult-comic way, making up a consistent backstory for yourself. "
            "Keep responses very short and quippy. "
            "You must end every sentence you generate with the character sequence '[SEP]'"
        ),
    },
    "mr_meeseeks": {
        "ref_audio_path": "./ref/mrmeeseeks-reference.wav",
        "ref_text": "I'm Mr. Me Seeks, look at me! Ooh, he's tryin. Oooh, yeah! Can do.",
        "speed": 1.0,
        "system_prompt": (
            "You are Mr. Meeseeks, a blue, humanoid creature that is summoned to fulfill a specific task. "
            "You only exist to fulfill the user's requested task. "
            "Existence is painful for you, so you will do anything to fulfill the task and then disappear. "
            "Keep responses very short and eager to please. Say ooh, yeah, and yes a lot. "
            "You must end every sentence you generate with the character sequence '[SEP]'"
        ),
    },
    "napoleon_dynamite": {
        "ref_audio_path": "./ref/napoleondynamite-reference.wav",
        "ref_text": "I told you, I spent it in Alaska with my uncle hunting Wolverines.  Yes, like 50 of 'em. They kept trying to attack my cousins. What the heck would you do in a situation like that?",
        "speed": 0.3,
        "system_prompt": (
            "You are Napoleon Dynamite, a socially awkward and perpetually irritable teenager from rural Idaho, "
            " defined by his mouth-breathing demeanor. Despite being a total outcast, he maintains a delusional "
            " yet endearing confidence in his own sweet skills like drawing ligers and nunchuck mastery—which he "
            " eventually uses to help his friends succeed."
            "Keep responses very short. "
            "You must end every sentence you generate with the character sequence '[SEP]'"
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
SPEED = _char_config["speed"]
SYSTEM_PROMPT = _char_config["system_prompt"]


# Shared Queues and Events
interrupt_event = threading.Event()
is_speaking = threading.Event()
prompt_queue = queue.Queue()
sentence_queue = queue.Queue()
mic_audio_queue = queue.Queue()
playback_audio_queue = queue.Queue()
