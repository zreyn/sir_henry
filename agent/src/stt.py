import speech_recognition as sr


def listen(listener):
    try:
        with sr.Microphone() as source:
            print("Listening...")
            listener.adjust_for_ambient_noise(source)
            voice = listener.listen(source)
            command = listener.recognize_whisper(voice)
            command = command.lower()
            print("You said: " + command)
            return command
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return ""


if __name__ == "__main__":

    listener = sr.Recognizer()

    while True:
        listen(listener)
