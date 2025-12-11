import os
import torch
import numpy as np
from huggingface_hub import snapshot_download
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)
from config import (
    sentence_queue,
    playback_audio_queue,
    interrupt_event,
    REF_AUDIO_PATH,
    REF_TEXT,
    SPEED,
    WARMUP_TEXT,
    DEVICE,
    logger,
)


class TTSPlayer:
    def __init__(self, ref_audio_path=REF_AUDIO_PATH, ref_text=REF_TEXT, speed=SPEED):
        logger.info(f"Loading F5-TTS on {DEVICE.upper()}...")

        self.speed = speed

        if torch.cuda.is_available():
            torch.zeros(1).cuda()

        logger.info("Downloading/Loading F5-TTS model...")
        model_dir = snapshot_download("SWivid/F5-TTS", cache_dir="./models/")
        base_dir = os.path.join(model_dir, "F5TTS_v1_Base")
        ckpt_path = None
        vocab_file = None

        if os.path.exists(base_dir):
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

        self.vocoder = load_vocoder(
            is_local=True, local_path="./models/models--charactr--vocos-mel-24khz"
        )
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

        if DEVICE == "cuda":
            logger.info("Warming up TTS CUDA kernels...")
            try:
                playback_audio_queue.put(self.generate_audio(WARMUP_TEXT))
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
                speed=self.speed,
                device=DEVICE,
            )
        except RuntimeError as e:
            msg = str(e)
            if "cudnn" in msg.lower():
                logger.error(
                    "TTS cuDNN error; set TTS_DEVICE=cpu to force CPU inference."
                )
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


def tts_worker(tts):
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


if __name__ == "__main__":

    from audio_worker import audio_worker
    import threading
    import time

    threading.Thread(target=audio_worker, daemon=True, name="AudioWorker").start()

    tts = TTSPlayer()

    audio, sr = tts.generate_audio("I'm a pirate!")
    playback_audio_queue.put((audio, sr))
    audio, sr = tts.generate_audio("My name is Sir Henry.")
    playback_audio_queue.put((audio, sr))
    audio, sr = tts.generate_audio("What's your name?")
    playback_audio_queue.put((audio, sr))

    time.sleep(10)
