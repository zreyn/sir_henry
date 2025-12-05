import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import sys
from huggingface_hub import snapshot_download
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

REF_AUDIO_PATH = "./models/sirhenry-reference.wav"  # <--- PUT YOUR 10s WAV FILE HERE
REF_TEXT = "Arr, ye callin' upon the spirit o' Sir Henry the Dread Pirate Roberts -- or what's left o` me, anyway. Aye, `tis true, the terror o` the Seven"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        self.vocoder = load_vocoder(
            is_local=False
        )  # Downloads from HuggingFace automatically

        # 3. Load the Main Model (DiT)
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
            ),
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",  # or 'bigvgan' if you use that
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=DEVICE,
        )

        # 3. Process Reference Audio (Voice Clone target)
        if not os.path.exists(ref_audio_path):
            print(f"Warning: Reference file '{ref_audio_path}' not found.")
            print(
                "Please create a 'reference.wav' file (5-10s of speech) for voice cloning."
            )
            self.ref_audio, self.ref_text = None, ""
        else:
            print(f"Loading reference voice: {ref_audio_path}")
            # This utility handles resampling and trimming
            self.ref_audio, self.ref_text = preprocess_ref_audio_text(
                ref_audio_path, ref_text
            )

    def speak(self, text):
        """
        Synthesizes text to audio and plays it immediately.
        """
        if not text.strip():
            return

        if self.ref_audio is None:
            print("Error: No reference audio loaded. Cannot generate speech.")
            return

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
            device=DEVICE,
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

        # Play audio using SoundDevice
        # audio is typically (channels, samples) or (samples,), we ensure it's correct for sd.play
        print("Playing...")
        sd.play(audio, sample_rate)
        sd.wait()  # Block until finished playing


def main():
    # Create the player (loads model once)
    tts = TTSPlayer()

    print("\n--- F5-TTS Standalone Tester ---")
    print(f"Reference Audio: {REF_AUDIO_PATH}")
    print("Type text to verify voice cloning. Type 'exit' to quit.\n")

    while True:
        try:
            text = input("Text to speak: ")
            if text.lower() in ["exit", "quit"]:
                break

            tts.speak(text)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
