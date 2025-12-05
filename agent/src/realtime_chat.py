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
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

# STT config
SAMPLE_RATE = 16000  # Silero VAD requires 16k
CHUNK_SIZE = 512  # Silero VAD requires 512, 1024, or 1536 samples
VAD_THRESHOLD = 0.5  # Confidence threshold for speech (0.0 to 1.0)
PAUSE_LIMIT = 0.8  # Seconds of silence to trigger end-of-speech
PRE_ROLL_MS = 200  # Milliseconds of audio to keep before speech starts (prevents cutting off first syllable)c

# TTS config
REF_AUDIO_PATH = "./models/sirhenry-reference.wav" # <--- PUT YOUR 10s WAV FILE HERE
REF_TEXT = "Arr, ye callin' upon the spirit o' Sir Henry the Dread Pirate Roberts -- or what's left o` me, anyway. Aye, `tis true, the terror o` the Seven"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



interrupt_event = threading.Event()
prompt_queue = queue.Queue()          # finalized user text
sentence_queue = queue.Queue()        # LLM sentences to TTS
mic_audio_queue = queue.Queue()       # raw mic chunks
playback_audio_queue = queue.Queue()  # audio chunks to play


def load_models():
    """
    Loads Silero VAD and Faster Whisper models.
    """
    print("Loading Silero VAD...")
    # Using 'onnx=False' for the PyTorch JIT version which is simple to use
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )

    print("Loading Faster Whisper (this may take a moment)...")
    # Automatically selects CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Use 'tiny' or 'small' for real-time speed. 'medium' or 'large' for accuracy.
    whisper_model = WhisperModel("small", device=device, compute_type=compute_type)

    print(f"Models loaded. Running on {device.upper()}.")
    return vad_model, whisper_model


def audio_callback(indata, frames, time, status):
    """
    Callback for sounddevice.
    This is called rapidly by the audio thread.
    We just copy the data to a thread-safe queue.
    """
    if status:
        print(status, file=sys.stderr)
    mic_audio_queue.put(indata.copy())


def is_speech(model, audio_chunk_int16):
    """
    Runs a single chunk of audio through the VAD model.
    """
    # Convert INT16 to FLOAT32 and Normalize (-1 to 1)
    audio_float32 = audio_chunk_int16.astype(np.float32) / 32768.0

    # Convert to Tensor
    audio_tensor = torch.from_numpy(audio_float32)

    # Check dimensionality (Silero expects 1D or 2D)
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.squeeze()

    # Get speech probability
    with torch.no_grad():
        speech_prob = model(audio_tensor, SAMPLE_RATE).item()

    return speech_prob > VAD_THRESHOLD


def listen():
    vad_model, whisper_model = load_models()

    # Pre-roll buffer (deque would be better, but list is fine for simple logic)
    pre_roll_chunks = int((PRE_ROLL_MS / 1000) * (SAMPLE_RATE / CHUNK_SIZE))
    pre_roll_buffer = []

    # State variables
    triggered = False
    speech_buffer = []
    silence_counter = 0
    silence_limit_chunks = int(PAUSE_LIMIT * (SAMPLE_RATE / CHUNK_SIZE))

    print("\n--- Listening... (Press Ctrl+C to stop) ---")

    # Start Microphone Stream
    # We record in INT16 to save memory bandwidth, then convert to FLOAT32 for AI
    with sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    ):

        while True:
            try:
                # 1. Get audio chunk
                chunk = mic_audio_queue.get()

                # 2. Check VAD
                # Note: We must feed chunks of 512/1024/1536 to Silero
                speech_detected = is_speech(vad_model, chunk)

                # 3. State Machine
                if triggered:
                    # WE ARE IN A SPEECH SEGMENT
                    speech_buffer.append(chunk)

                    if speech_detected:
                        silence_counter = 0
                        sys.stdout.write("!")  # Visual indicator of speech
                        sys.stdout.flush()
                    else:
                        silence_counter += 1
                        sys.stdout.write(".")  # Visual indicator of silence
                        sys.stdout.flush()

                    # End of speech detected?
                    if silence_counter >= silence_limit_chunks:
                        print(f"\n[Processing {len(speech_buffer)} chunks...]")

                        # --- TRANSCRIBE ---
                        # Combine all chunks into one float32 array
                        full_audio = np.concatenate(speech_buffer)
                        full_audio_float = full_audio.astype(np.float32) / 32768.0

                        # Ensure 1D array (Whisper expects mono audio as 1D)
                        if len(full_audio_float.shape) > 1:
                            full_audio_float = full_audio_float.squeeze()

                        # Limit audio length to prevent OOM (30 seconds max)
                        max_samples = int(30 * SAMPLE_RATE)
                        if len(full_audio_float) > max_samples:
                            print(
                                f"\n[Warning: Audio truncated from {len(full_audio_float)/SAMPLE_RATE:.1f}s to 30s]"
                            )
                            full_audio_float = full_audio_float[:max_samples]

                        # Run Whisper with memory-efficient settings
                        # beam_size=1 uses greedy decoding (much lower memory)
                        # best_of=1 further reduces memory usage
                        print(f"\n[running whisper...]")
                        try:
                            segments, info = whisper_model.transcribe(
                                full_audio_float,
                                beam_size=1,  # Greedy decoding (lower memory)
                                best_of=1,  # No sampling (lower memory)
                                language=None,  # Auto-detect
                                task="transcribe",
                                vad_filter=True,  # Use built-in VAD to skip silence
                            )

                            text = "".join([segment.text for segment in segments])

                            if text.strip():
                                print(f"\nUser: {text.strip()}")
                            else:
                                print("\n(No text detected)")
                        except RuntimeError as e:
                            if (
                                "out of memory" in str(e).lower()
                                or "oom" in str(e).lower()
                            ):
                                print(
                                    f"\n[Error: Out of memory. Audio too long ({len(full_audio_float)/SAMPLE_RATE:.1f}s). Try speaking shorter segments.]"
                                )
                            else:
                                print(f"\n[Error during transcription: {e}]")
                        except Exception as e:
                            print(f"\n[Error during transcription: {e}]")

                        # Reset State
                        triggered = False
                        speech_buffer = []
                        silence_counter = 0
                        pre_roll_buffer = []
                        print("\n--- Listening... ---")

                else:
                    # WE ARE LISTENING FOR SPEECH
                    if speech_detected:
                        print("\n[Speech Detected]")
                        triggered = True
                        # Add the pre-roll context so we don't cut off the first word
                        speech_buffer.extend(pre_roll_buffer)
                        speech_buffer.append(chunk)
                        silence_counter = 0
                    else:
                        # Manage Pre-roll buffer (keep it a fixed size)
                        pre_roll_buffer.append(chunk)
                        if len(pre_roll_buffer) > pre_roll_chunks:
                            pre_roll_buffer.pop(0)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                break

    print("Stopped.")

def mic_worker():
    listen()

def llm_worker():
    while True:
        user_text = prompt_queue.get()
        interrupt_event.clear()

        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": user_text, "stream": True},
            stream=True,
        )
        buffer = ""
        for line in resp.iter_lines():
            if interrupt_event.is_set():
                break
            if not line:
                continue
            # Ollama streams JSON per line
            data = json.loads(line)
            token = data.get("response", "")
            buffer += token
            if re.search(r"[\\.\\!\\?\\n]$", token):
                sentence_queue.put(buffer.strip())
                buffer = ""

class TTSPlayer:
    def __init__(self, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT):
        print(f"Loading F5-TTS on {DEVICE.upper()}...")
        
        # 1. Download model from HuggingFace if not already cached
        print("Downloading F5-TTS model from HuggingFace...")
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
        
        print(f"Using checkpoint: {ckpt_path}")
        
        # 2. Load the Vocoder (converts spectrogram to audio)
        self.vocoder = load_vocoder(is_local=False) # Downloads from HuggingFace automatically
        
        # 3. Load the Main Model (DiT)
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",  # or 'bigvgan' if you use that
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=DEVICE
        )
        
        # 3. Process Reference Audio (Voice Clone target)
        if not os.path.exists(ref_audio_path):
            print(f"Warning: Reference file '{ref_audio_path}' not found.")
            print("Please create a 'reference.wav' file (5-10s of speech) for voice cloning.")
            self.ref_audio, self.ref_text = None, ""
        else:
            print(f"Loading reference voice: {ref_audio_path}")
            # This utility handles resampling and trimming
            self.ref_audio, self.ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

    def generate_audio(self, text):
        """
        Synthesizes text to audio and returns (audio, sample_rate).
        """
        if not text.strip():
            return None, None

        if self.ref_audio is None:
            print("Error: No reference audio loaded. Cannot generate speech.")
            return None, None

        print(f"Generating: '{text}'...")
        
        # F5-TTS Inference
        # Returns: audio_wave, sample_rate, spectrogram
        audio, sample_rate, _ = infer_process(
            self.ref_audio, 
            self.ref_text, 
            text, 
            self.model, 
            self.vocoder, 
            mel_spec_type="vocos",
            speed=1.0,
            device=DEVICE
        )
        
        # Ensure correct dtype/shape and tame peaks
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.squeeze(audio)
        peak = np.max(np.abs(audio)) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
            print(f"(normalized audio, peak was {peak:.3f})")
        audio = np.clip(audio * 0.9, -1.0, 1.0)  # slight attenuation to avoid clipping

        # Save to file for inspection
        try:
            sf.write("last_output.wav", audio, sample_rate)
            print("Saved to last_output.wav")
        except Exception as e:
            print(f"(warning) could not save wav: {e}")

        return audio, sample_rate

def tts_worker():
    while True:
        text = sentence_queue.get()
        if interrupt_event.is_set():
            continue
        audio, sr = tts.generate_audio(text)
        if audio is None or sr is None:
            continue
        playback_audio_queue.put((audio, sr))

def stop_audio_playback():
    sd.stop()

def audio_worker():
    while True:
        audio, sr = playback_audio_queue.get()
        if interrupt_event.is_set():
            continue
        sd.play(audio, sr)
        sd.wait()

if __name__ == "__main__":
    tts = TTSPlayer()
    threading.Thread(target=mic_worker, daemon=True).start()
    threading.Thread(target=llm_worker, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=audio_worker, daemon=True).start()
    print("Live loop running. Speak to interact. Ctrl+C to exit.")
    threading.Event().wait()  # keep main alive
