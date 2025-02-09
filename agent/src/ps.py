import sounddevice as sd
import numpy as np
import speech_recognition as sr


def get_default_input_device():
    """Prints information about the default input device."""
    default_input_device_index = sd.default.device["input"]  # Get the index of the default input device
    devices = sd.query_devices()  # Get a list of all devices
    return devices[default_input_device_index]  # Get the device info


def read_from_mic(mic, sample_rate=16000, duration=5):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), dtype='int16', samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    print("Playing back audio...")
    # Play back the recorded audio
    sd.play(audio_data, sample_rate)
    sd.wait()  # Wait for playback to finish
    print("Playback finished.")

    audio = sr.AudioData(audio_data.tobytes(), sample_rate, 2)  # Assuming 2 bytes per sample (16-bit)

    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_sphinx(audio)
        print("Sphinx thinks you said: " + text)
        return text
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))
        return ""


if __name__ == "__main__":
    
    mic = get_default_input_device()
    print(f"Reading from {mic['name']}... get ready")
    
    result = read_from_mic(mic)
    print(result)

