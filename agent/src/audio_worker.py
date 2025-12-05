import queue
import sounddevice as sd
from config import (
    playback_audio_queue,
    interrupt_event,
    is_speaking,
    logger,
)


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
