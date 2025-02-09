import whisper
import warnings

import sounddevice as sd
import numpy as np


def get_default_input_device():
    """Prints information about the default input device."""
    default_input_device_index = sd.default.device["input"]  # Get the index of the default input device
    devices = sd.query_devices()  # Get a list of all devices
    return devices[default_input_device_index]  # Get the device info


def read_from_file(model, filename):
    result = model.transcribe(filename)  # Transcribe an audio file
    return result["text"]


def read_from_mic(mic, model, sample_rate=44100, duration=3):
    print("Recording...")
    fs = mic["default_samplerate"] # Sample rate
    myrecording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    audio_data = (
        myrecording.flatten().astype(np.float32) / np.finfo(myrecording.dtype).max
    )  # Normalize
    audio = whisper.pad_or_trim(audio_data, whisper.audio.SAMPLE_RATE)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    print("transcribing")
    result = model.transcribe(mel, language='en')
    print(result)
    return result["text"]


if __name__ == "__main__":
    print("Loading STT model")

    warnings.filterwarnings(
        "ignore", message="FP16 is not supported on CPU; using FP32 instead"
    )
    model = whisper.load_model(
        "base"
    )  # Load a model (tiny, base, small, medium, large)

    # print("Reading from file")
    # result = read_from_file(model, "test2.wav")
    # print(result)

    mic = get_default_input_device()
    print(f"Reading from {mic['name']}... get ready")
    result = read_from_mic(mic, model)
    print(result)

