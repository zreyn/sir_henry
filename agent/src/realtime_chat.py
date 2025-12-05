import queue
import threading
import requests
import json
import re
import time
import os
import sys
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import logging
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(threadName)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# REMOVED from top level to prevent CTranslate2/Torch conflict:
# from faster_whisper import WhisperModel

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

# STT config
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
VAD_THRESHOLD = 0.5
PAUSE_LIMIT = 0.8
PRE_ROLL_MS = 200

# TTS config
REF_AUDIO_PATH = "./models/sirhenry-reference.wav"
REF_TEXT = "Arr, ye callin' upon the spirit o' Sir Henry the Dread Pirate Roberts -- or what's left o` me, anyway."
_FORCE_DEV = os.environ.get("TTS_DEVICE", "").lower()
DEVICE = (
    "cpu" if _FORCE_DEV == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
)

SYSTEM_PROMPT = (
    "You are a dead 15th century pirate named Sir Henry, now only a skeleton still wearing pirate clothes. "
    "You were once the Dread Pirate Roberts, but you slipped on a banana peel and fell overboard to your death. "
    "Your favorite snack is spare ribs. "
    "Converse with live, modern-day people in an insult-comic way, making up a consistent backstory for yourself. "
    "Keep most of your responses short. "
)

interrupt_event = threading.Event()
is_speaking = threading.Event()
prompt_queue = queue.Queue()
sentence_queue = queue.Queue()
mic_audio_queue = queue.Queue()
playback_audio_queue = queue.Queue()


def load_models():
    """
    Loads Silero VAD and Faster Whisper models.
    """
    logger.info("Loading Silero VAD...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )

    logger.info("Loading Faster Whisper...")
    # --- LAZY IMPORT FIX ---
    # We import here to ensure PyTorch (F5-TTS) has already loaded its CUDA libraries
    # in the main thread before CTranslate2 (Whisper) loads its libraries.
    from faster_whisper import WhisperModel

    # Allow overriding STT device via env var, default to cuda if available
    # stt_device =
    stt_device = "cpu"
    compute_type = "float16" if stt_device == "cuda" else "int8"

    logger.info(f"Initializing Whisper on {stt_device.upper()}...")
    whisper_model = WhisperModel("small", device=stt_device, compute_type=compute_type)

    logger.info(f"Models loaded. STT running on {stt_device.upper()}.")
    return vad_model, whisper_model


def audio_callback(indata, frames, time, status):
    if status:
        logger.error(f"Audio callback status: {status}")
    mic_audio_queue.put(indata.copy())


def is_speech(model, audio_chunk_int16):
    audio_float32 = audio_chunk_int16.astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(audio_float32)
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.squeeze()
    with torch.no_grad():
        speech_prob = model(audio_tensor, SAMPLE_RATE).item()
    return speech_prob > VAD_THRESHOLD


def listen():
    # This will now trigger the local import of Faster Whisper
    vad_model, whisper_model = load_models()

    pre_roll_chunks = int((PRE_ROLL_MS / 1000) * (SAMPLE_RATE / CHUNK_SIZE))
    pre_roll_buffer = []

    triggered = False
    speech_buffer = []
    silence_counter = 0
    silence_limit_chunks = int(PAUSE_LIMIT * (SAMPLE_RATE / CHUNK_SIZE))

    logger.info("Listening... (Press Ctrl+C to stop)")

    with sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    ):
        while True:
            try:
                chunk = mic_audio_queue.get()
                speech_detected = is_speech(vad_model, chunk)

                if triggered:
                    speech_buffer.append(chunk)
                    if speech_detected:
                        silence_counter = 0
                        # Removed visual feedback for cleaner logs
                    else:
                        silence_counter += 1

                    if silence_counter >= silence_limit_chunks:
                        logger.info(f"Processing {len(speech_buffer)} chunks...")

                        # Stop audio playback if user interrupts
                        if not playback_audio_queue.empty() or is_speaking.is_set():
                            interrupt_event.set()
                            # Clear queues to stop old bot speech
                            with sentence_queue.mutex:
                                sentence_queue.queue.clear()
                            with playback_audio_queue.mutex:
                                playback_audio_queue.queue.clear()
                            try:
                                # Attempt to stop playback if running
                                sd.stop()
                            except Exception:
                                pass
                            time.sleep(0.1)  # Give threads time to notice interrupt

                        full_audio = np.concatenate(speech_buffer)
                        full_audio_float = full_audio.astype(np.float32) / 32768.0

                        if len(full_audio_float.shape) > 1:
                            full_audio_float = full_audio_float.squeeze()

                        # Run Whisper
                        try:
                            segments, info = whisper_model.transcribe(
                                full_audio_float,
                                beam_size=1,
                                best_of=1,
                                language=None,
                                task="transcribe",
                                vad_filter=True,
                            )
                            text = "".join([segment.text for segment in segments])
                            cleaned = text.strip()
                            if cleaned:
                                logger.info(f"User: {cleaned}")
                                prompt_queue.put(cleaned)
                            else:
                                logger.info("(No text detected)")
                        except Exception as e:
                            logger.error(f"Error during transcription: {e}")

                        triggered = False
                        speech_buffer = []
                        silence_counter = 0
                        pre_roll_buffer = []
                        # logger.info("Listening...") # Avoid spamming "Listening..." after every turn
                else:
                    if speech_detected:
                        logger.info("Speech Detected")
                        triggered = True
                        speech_buffer.extend(pre_roll_buffer)
                        speech_buffer.append(chunk)
                        silence_counter = 0
                    else:
                        pre_roll_buffer.append(chunk)
                        if len(pre_roll_buffer) > pre_roll_chunks:
                            pre_roll_buffer.pop(0)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                break


def mic_worker():
    listen()


def llm_worker():
    while True:
        user_text = prompt_queue.get()
        # Clear interrupt flag when we start a new turn
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


class TTSPlayer:
    def __init__(self, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT):
        logger.info(f"Loading F5-TTS on {DEVICE.upper()}...")

        # Ensure PyTorch has initialized CUDA before anything else happens
        if torch.cuda.is_available():
            torch.zeros(1).cuda()

        logger.info("Downloading/Loading F5-TTS model...")
        model_dir = snapshot_download("SWivid/F5-TTS", cache_dir="./models/")

        # Use the F5TTS_v1_Base model checkpoint
        # Check for safetensors first (preferred), then .pt file
        base_dir = os.path.join(model_dir, "F5TTS_v1_Base")
        ckpt_path = None
        vocab_file = None

        if os.path.exists(base_dir):
            # Try safetensors first
            safetensors_path = os.path.join(base_dir, "model_1250000.safetensors")
            pt_path = os.path.join(base_dir, "model_1250000.pt")
            vocab_path = os.path.join(base_dir, "vocab.txt")

            if os.path.exists(safetensors_path):
                ckpt_path = safetensors_path
            elif os.path.exists(pt_path):
                ckpt_path = pt_path
            else:
                raise FileNotFoundError(f"No checkpoint found in {base_dir}")

            if os.path.exists(vocab_path):
                vocab_file = vocab_path
        else:
            raise FileNotFoundError(f"Model directory not found: {base_dir}")

        logger.info(f"Using checkpoint: {ckpt_path}")

        self.vocoder = load_vocoder(is_local=False)
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
            ),
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=DEVICE,
        )

        if not os.path.exists(ref_audio_path):
            logger.warning(f"Reference file '{ref_audio_path}' not found.")
            self.ref_audio, self.ref_text = None, ""
        else:
            logger.info(f"Loading reference voice: {ref_audio_path}")
            self.ref_audio, self.ref_text = preprocess_ref_audio_text(
                ref_audio_path, ref_text
            )

        # Warmup to ensure CUDA kernels are loaded before other libraries (like CTranslate2) interfere
        if DEVICE == "cuda":
            logger.info("Warming up TTS CUDA kernels...")
            try:
                self.generate_audio("Warmup.")
                logger.info("TTS Warmup successful.")
            except Exception as e:
                logger.error(f"TTS Warmup failed: {e}")
                logger.info(
                    "Suggestion: Try running with STT_DEVICE=cpu to avoid library conflicts."
                )

    def generate_audio(self, text):
        if not text.strip():
            return None, None
        if self.ref_audio is None:
            return None, None

        logger.info(f"Generating: '{text}'...")
        try:
            audio, sample_rate, _ = infer_process(
                self.ref_audio,
                self.ref_text,
                text,
                self.model,
                self.vocoder,
                mel_spec_type="vocos",
                speed=1.0,
                device=DEVICE,
            )
        except RuntimeError as e:
            msg = str(e)
            if "cudnn" in msg.lower():
                logger.error("TTS cuDNN error; set TTS_DEVICE=cpu to force CPU inference.")
            else:
                logger.error(f"TTS inference error: {e}")
            return None, None

        # Normalization logic
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.squeeze(audio)
        # Simple normalization to -1.0 to 1.0
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio, sample_rate


def tts_worker():
    while True:
        text = sentence_queue.get()
        if interrupt_event.is_set():
            continue

        try:
            audio, sr = tts.generate_audio(text)
            if audio is None:
                continue
            playback_audio_queue.put((audio, sr))
        except Exception as e:
            logger.error(f"TTS Error: {e}")


def audio_worker():
    while True:
        audio, sr = playback_audio_queue.get()
        if interrupt_event.is_set():
            # Drain queue if interrupted
            while not playback_audio_queue.empty():
                try:
                    playback_audio_queue.get_nowait()
                except queue.Empty:
                    break
            continue

        try:
            is_speaking.set()
            sd.play(audio, sr)
            sd.wait()
        except Exception as e:
            logger.error(f"Audio Playback Error: {e}")
        finally:
            is_speaking.clear()


if __name__ == "__main__":
    # Force PyTorch CUDA initialization FIRST
    if torch.cuda.is_available():
        logger.info("Initializing PyTorch CUDA context...")
        torch.zeros(1).cuda()

    # Initialize TTS (which uses PyTorch) BEFORE importing Faster Whisper in the threads
    try:
        tts = TTSPlayer()
    except Exception as e:
        logger.error(f"Failed to load TTS: {e}")
        sys.exit(1)

    # Start threads
    threading.Thread(target=mic_worker, daemon=True, name="MicWorker").start()
    threading.Thread(target=llm_worker, daemon=True, name="LLMWorker").start()
    threading.Thread(target=tts_worker, daemon=True, name="TTSWorker").start()
    threading.Thread(target=audio_worker, daemon=True, name="AudioWorker").start()

    logger.info("System Ready. Speak to interact.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        # Set interrupt event to stop workers
        interrupt_event.set()
        
        # Force stop any playing audio
        try:
            sd.stop()
        except:
            pass
            
        sys.exit(0)
