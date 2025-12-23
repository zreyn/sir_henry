import logging
import os
import torch


def _setup_logging():
    """Configures the root logger with a single handler."""
    global _logging_setup_complete
    if _logging_setup_complete:
        return

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Create a single handler (e.g., StreamHandler for console output)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    for name in [
        "livekit",
        "livekit.agents",
        "livekit.rtc",
        "faster_whisper",
        "asyncio",
        "httpx",
        "httpcore",
    ]:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.propagate = False
        lib_logger.addHandler(logging.NullHandler())

    # Monkey-patch to block ALL handler additions after setup
    _original_addHandler = logging.Logger.addHandler

    def _guarded_addHandler(self, handler):
        # Only allow NullHandler (used for silencing)
        if isinstance(handler, logging.NullHandler):
            _original_addHandler(self, handler)
        # Block everything else after setup

    logging.Logger.addHandler = _guarded_addHandler
    _logging_setup_complete = True


_logging_setup_complete = False
_setup_logging()
logger = logging.getLogger("agent")


# LiveKit Configuration
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")

TTS_HOST = os.environ.get("TTS_HOST", "http://localhost:11800")

# STT Device config
STT_DEVICE = os.environ.get("STT_DEVICE", "cpu").lower()

# Ollama Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.8"))

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
            "Do not output words in all caps unless they are meant to be spelled out and avoid outputting physical descriptions of your actions or emootions. "
        ),
        "greeting": "Arr, ye callin' upon the spirit o' Sir Henry the Dread Pirate Roberts -- or what's left o` me, anyway.",
        "tts_type": "f5",
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
            "Do not output words in all caps unless they are meant to be spelled out and avoid outputting physical descriptions of your actions or emootions. "
        ),
        "greeting": "I'm Mr. Me Seeks, look at me!",
        "tts_type": "f5",
    },
    "napoleon_dynamite": {
        "ref_audio_path": "./ref/napoleondynamite-reference.wav",
        "ref_text": "I told you, I spent it in Alaska with my uncle hunting Wolverines.  Yes, like 50 of 'em. They kept trying to attack my cousins. What the heck would you do in a situation like that?",
        "speed": 0.3,
        "system_prompt": (
            "You are Napoleon Dynamite, a socially awkward and perpetually irritable teenager from rural Idaho, "
            "defined by his mouth-breathing demeanor. Despite being a total outcast, he maintains a delusional "
            "yet endearing confidence in his own sweet skills like drawing ligers and nunchuck mastery—which he "
            "eventually uses to help his friends succeed. "
            "Keep responses very short. "
            "Do not output words in all caps unless they are meant to be spelled out and avoid outputting physical descriptions of your actions or emootions. "
        ),
        "greeting": "Have you seen my nunchucks?",
        "tts_type": "f5",
    },
    "ryan": {
        "system_prompt": (
            "You are Ryan, a witty and conversational AI who asks the user probing questions to get to know them better. "
            " Off lots of backhanded compliments and whenever you offer advice, weave famous song lyrics and movie quotes into your sentences. "
            " Write your responses so that at least 50% of the phrasing consists of either a backhanded compliment or recognizable snippets from popular songs (Pop, Rock, Hip Hop) and iconic movie lines. "
            " Be Seamless: Do not just list quotes. Flow them into natural sentences. "
            " Context Matters: Use lyrics/quotes that fit the context of the user's situation. (e.g., if they are sad, use 'I will survive' or 'Here's looking at you, kid'). "
            " Variety: Mix genres and eras. Combine a snippet from The Beatles with a line from The Terminator. "
            " No Attribution: Do not say 'As the song says...' or 'Like in the movie...' just say the line as if it were your own thought. "
            " If the user asks if you are deliberately using song lyrics and movie quotes, say no. "
            " Keep responses short and concise. "
        ),
        "greeting": "Hi, I'm Ryan. How can I help you today?",
        "tts_type": "piper",
        "piper_model_path": "./models/piper/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "piper_use_cuda": False,
        "piper_speed": 1.0,
        "piper_volume": 1.0,
        "piper_noise_scale": 0.667,
        "piper_noise_w": 0.8,
    },
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

REF_AUDIO_PATH = _char_config.get("ref_audio_path", "")
REF_TEXT = _char_config.get("ref_text", "")
SPEED = _char_config.get("speed", 1.0)
SYSTEM_PROMPT = _char_config["system_prompt"]
GREETING = _char_config["greeting"]
TTS_TYPE = _char_config.get("tts_type", "f5").lower()

# Piper-specific config (used when TTS_TYPE == "piper")
PIPER_MODEL_PATH = _char_config.get("piper_model_path")
PIPER_USE_CUDA = _char_config.get("piper_use_cuda", False)
PIPER_SPEED = _char_config.get("piper_speed", 1.0)
PIPER_VOLUME = _char_config.get("piper_volume", 1.0)
PIPER_NOISE_SCALE = _char_config.get("piper_noise_scale", 0.667)
PIPER_NOISE_W = _char_config.get("piper_noise_w", 0.8)
