import torch
import numpy as np
import sounddevice as sd
import queue
import time
import sys
from faster_whisper import WhisperModel

# --- Configuration ---
SAMPLE_RATE = 16000  # Silero VAD requires 16k
CHUNK_SIZE = 512  # Silero VAD requires 512, 1024, or 1536 samples
VAD_THRESHOLD = 0.5  # Confidence threshold for speech (0.0 to 1.0)
PAUSE_LIMIT = 0.8  # Seconds of silence to trigger end-of-speech
PRE_ROLL_MS = 200  # Milliseconds of audio to keep before speech starts (prevents cutting off first syllable)

# --- Global Queues ---
audio_queue = queue.Queue()


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
    audio_queue.put(indata.copy())


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
                chunk = audio_queue.get()

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


if __name__ == "__main__":
    listen()
