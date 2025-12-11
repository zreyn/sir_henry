import sys
import time
import torch
import numpy as np
import sounddevice as sd
from config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    VAD_THRESHOLD,
    PAUSE_LIMIT,
    PRE_ROLL_MS,
    interrupt_event,
    is_speaking,
    prompt_queue,
    sentence_queue,
    playback_audio_queue,
    mic_audio_queue,
    logger,
)


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

                # Ignore microphone input when system is speaking to prevent feedback loop
                if is_speaking.is_set():
                    continue

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


def stt_worker():
    listen()
