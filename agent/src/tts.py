import pyttsx3


def setup_tts():
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 150)
    tts_engine.setProperty("volume", 1.0)
    # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Bahh") # shaky, gollum voice
    # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Whisper") # creepy whisper
    # tts_engine.setProperty("voice", "com.apple.eloquence.en-US.Rocko") # generic war games voice
    # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Albert") # creepy, strained voice
    # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Boing") # robot goblin
    # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Bubbles") # sounds underwater
    # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Cellos") # ominous cello singing
    # tts_engine.setProperty("voice", "com.apple.speech.synthesis.voice.Deranged")  # shaky, crazy voice
    tts_engine.setProperty("voice", "com.apple.voice.compact.en-GB.Daniel")

    return tts_engine


def text_to_speech(text, tts_engine):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error during TTS conversion: {e}")


if __name__ == "__main__":
    print("Setting up TTS engine...")
    tts_engine = setup_tts()

    print("Type what you want to hear")
    while True:
        user_input = input("> ").lower()

        if user_input == "end":
            print("Bye!")
            break

        text_to_speech(user_input, tts_engine)
