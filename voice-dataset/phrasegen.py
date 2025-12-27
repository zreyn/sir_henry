import os
import requests
from dotenv import load_dotenv
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phrasegen")

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


def setup_ollama():
    return [
        {
            "role": "system",
            "content": """
You are a linguistic data specialist creating a dataset for Text-to-Speech (TTS) fine-tuning.
Your goal is to generate text that is phonetically diverse, grammatically complex, and varies in sentiment and intonation.
""",
        }
    ]


def chat_ollama(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "num_predict": 256,
            "repeat_penalty": 1.15,
            "frequency_penalty": 0.6,
            "presence_penalty": 0.6,
        },
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


def save_phrases(phrases, phrase_file_name):
    with open(phrase_file_name, "a") as f:
        for phrase in phrases:
            f.write(phrase + "\n")


def delete_phrase_file(phrase_file_name):
    if os.path.exists(phrase_file_name):
        os.remove(phrase_file_name)
        return True
    return False


def main():

    logger.info("Setting up Ollama...")
    messages = setup_ollama()

    phrase_file_name = "raw/phrases.txt"
    deleted = delete_phrase_file(phrase_file_name)
    if deleted:
        logger.info(f"Old phrase file '{phrase_file_name}' has been deleted.")

    number_of_phrases_per_topic = 70
    topics = [
        'Pirate speak: treasure, the sea, ship life, fighting, drinking, and insults.',
        'Folklore & Mythology: mythical creatures, ancient legends, and supernatural phenomena.',
        'Puns and Wordplay, Irony and Sarcasm, Hyperbole and Understatement,Juxtaposition and Incongruity',
        'Allusion and Pop Culture References',
        'Sci-Fi & Space Exploration: Narrative sentences about starships, distant planets, and futuristic concepts.',
        'Children\'s Stories: Soft, slower, and explanatory tone.',
        'Argumentative/Debate: Persuasive text that uses "I believe," "However," and "Therefore" to capture firm intonation.',
        'Casual Texting/Slang: Short, punchy sentences with contractions ("I\'m gonna head out," "What\'s up?").',
        'The "Harvard Sentences" (Phonetic Balance): As mentioned before, standard phonetically balanced sentences to catch any missing sounds (e.g., "The small pup gnawed a hole in the sock").',
    ]

    logger.info(
        f"Generating {len(topics)} topics with phrases per topic {number_of_phrases_per_topic} [{number_of_phrases_per_topic * len(topics)} total phrases]"
    )

    for topic in topics:
        logger.info(f"Generating phrases for topic: {topic}")
        total_phrases = 0

        topic_prompt = f"""
        Generate {number_of_phrases_per_topic} unique sentences for a TTS training dataset. The sentences must strictly follow these constraints to ensure high-quality model alignment:
        1. Phonetic Diversity: Include a mix of sibilants (s, sh, z), plosives (p, b, t, d), and nasals. Use complex words occasionally.
        2. Structural Variety:
        {number_of_phrases_per_topic/5} sentences should be questions (Who, What, Where, Why, How).
        {number_of_phrases_per_topic/5} sentences should be short exclamations or commands.
        {number_of_phrases_per_topic/5} sentences should be long, complex compound sentences with multiple clauses.
        {number_of_phrases_per_topic/5} sentences must contain numbers, dates, or currency (written out as they should be spoken, e.g., "twenty-twenty five").
        {number_of_phrases_per_topic/5} sentences should be conversational/dialogue (using "I", "You", contractions).
        3. Punctuation: Use commas, colons, semi-colons, and ellipses (...) to force the TTS to learn pausing logic.
        4. Topic: Focus this batch on: {topic}.
        Output Format:
        Provide only the raw sentences, one per line. Do not number them. Do not include quotes around the sentences.
        """

        while total_phrases < number_of_phrases_per_topic:

            messages.append({"role": "user", "content": topic_prompt})

            try:
                reply = chat_ollama(messages)
            except Exception as e:
                print(f"(error calling Ollama: {e})")
                continue

            messages.append({"role": "assistant", "content": reply})

            new_phrases = len(reply.split('\n'))
            total_phrases += new_phrases
            logger.info(f"Model responded with {new_phrases} phrases")

            logger.info(f"Saving phrases")
            save_phrases(reply.split("\n"), phrase_file_name)




if __name__ == "__main__":
    main()
