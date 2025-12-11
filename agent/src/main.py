import time
import sys
import threading
import torch
import sounddevice as sd
from config import logger, interrupt_event
from stt_worker import stt_worker
from llm_worker import llm_worker
from tts_worker import tts_worker, TTSPlayer
from audio_worker import audio_worker


def main():
    if torch.cuda.is_available():
        logger.info("Initializing PyTorch CUDA context...")
        torch.zeros(1).cuda()

    try:
        tts = TTSPlayer()
    except Exception as e:
        logger.error(f"Failed to load TTS: {e}")
        sys.exit(1)

    threading.Thread(target=stt_worker, daemon=True, name="STTWorker").start()
    threading.Thread(target=llm_worker, daemon=True, name="LLMWorker").start()
    threading.Thread(
        target=tts_worker, args=(tts,), daemon=True, name="TTSWorker"
    ).start()
    threading.Thread(target=audio_worker, daemon=True, name="AudioWorker").start()

    logger.info("System Ready. Speak to interact.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        interrupt_event.set()
        try:
            sd.stop()
        except:
            pass

        sys.exit(0)


if __name__ == "__main__":
    main()
