import pyttsx3

engine = pyttsx3.init()

# Get and print information about the current voice
current_voice = engine.getProperty("voice")
print("Current Voice:")
print(current_voice)

# # List available voices and their IDs
# voices = engine.getProperty("voices")
# print("\nAvailable Voices:")
# for voice in voices:
#     if "en_US" in voice.languages:
#         print(f"ID: {voice.id}")
#         print(f"Name: {voice.name}")
#         print(f"Languages: {voice.languages}")
#         print(f"Gender: {voice.gender}")
#         print("-----")

#         engine.setProperty('voice', voice.id)
#         engine.say(f"I be Sir Henry, once the terror of the seven seas, with a crew as wild as the tempest itself.")
#         engine.runAndWait()


engine.setProperty("rate", 150)
engine.setProperty("volume", 0.8)
# engine.setProperty("voice", "com.apple.speech.synthesis.voice.Bahh") # shaky, gollum voice
# engine.setProperty("voice", "com.apple.speech.synthesis.voice.Whisper") # creepy whisper
# engine.setProperty("voice", "com.apple.eloquence.en-US.Rocko") # generic war games voice
# engine.setProperty("voice", "com.apple.speech.synthesis.voice.Albert") # creepy, strained voice
# engine.setProperty("voice", "com.apple.speech.synthesis.voice.Boing") # robot goblin
# engine.setProperty("voice", "com.apple.speech.synthesis.voice.Bubbles") # sounds underwater
# engine.setProperty("voice", "com.apple.speech.synthesis.voice.Cellos") # ominous cello singing
engine.setProperty("voice", "com.apple.speech.synthesis.voice.Deranged") # shaky, crazy voice

engine.say(
    # "Thousands of years ago, into the future, there was an army of monkeys..."
    "Arrr, gather 'round and lend me yer ears! I be Sir Henry, once the terror of the seven seas, with a crew as wild as the tempest itself. We sailed aboard the majestic Sea Raven, plunderin' treasure and seekin' adventure across the world. Alas, a cursed gold piece sent my soul to Davy Jones' locker, leavin' me as ye see now—mere bone but with stories aplenty!"
)
engine.runAndWait()
