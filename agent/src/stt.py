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
    audio_data = sd.rec(int(duration * fs), dtype='int16', samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    print("Playing back audio...")
    # Play back the recorded audio
    sd.play(audio_data, fs)
    sd.wait()  # Wait for playback to finish
    print("Playback finished.")

    print("Padding and spectrogramming...")
    audio = whisper.pad_or_trim(audio_data.flatten().astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    print("Transcribing...")
    result = model.transcribe(mel, language='en')
    print(result)
    return result["text"]


if __name__ == "__main__":
    print("Loading STT model")

    warnings.filterwarnings(
        "ignore", message="FP16 is not supported on CPU; using FP32 instead"
    )
    model = whisper.load_model("base")  # (tiny, base, small, medium, large)

    # print("Reading from file")
    # result = read_from_file(model, "test2.wav")
    # print(result)

    mic = get_default_input_device()
    print(f"Reading from {mic['name']}... get ready")
    result = read_from_mic(mic, model)
    print(result)




# import whisper
# import warnings
# import sounddevice as sd
# import numpy as np

# def get_default_input_device():
#     """Prints information about the default input device."""
#     default_input_device_index = sd.default.device["input"]  # Get the index of the default input device
#     devices = sd.query_devices()  # Get a list of all devices
#     return devices[default_input_device_index]  # Get the device info

# model = whisper.load_model("base")  # (tiny, base, small, medium, large)
# mic = get_default_input_device()
# fs = mic["default_samplerate"] # Sample rate
# duration = 3

# audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1) ; sd.wait()

# sd.play(audio_data, fs) ; sd.wait()

# audio = whisper.pad_or_trim(audio_data.flatten().astype(np.float32))
# mel = whisper.log_mel_spectrogram(audio).to(model.device)