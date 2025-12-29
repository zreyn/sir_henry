This module is used to create datasets for training custom voices.

The data is accumulated in the `raw` directory first, then converted for training. 

1. Generate phrases using the local ollama model that give at least 30 mins of audio samples:
```
uv sync
uv run python phrasegen.py
```

2. Get audio to go with the phrases.

    * Call ElevenLabs (for Sir Henry) for TTS samples and prep the dataset.
    ```
    uv run python tts_generator.py
    ``` 

    * For a new custom voice, I need to add something to show the phrases and record the wav files. 

3. Fine-tune f5-tts:
```
python src/f5_tts/train/datasets/prepare_csv_wavs.py <absolute-path>/sir_henry/voice-dataset/sir_henry <absolute-path>/F5-TTS/data/sir_henry_pinyin
```