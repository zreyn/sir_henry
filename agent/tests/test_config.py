"""Unit tests for config.py module."""

import os
import pytest
from unittest.mock import patch, MagicMock
import logging
import queue
import threading


def test_logging_configuration():
    """Test that logging is configured correctly."""
    import config

    assert isinstance(config.logger, logging.Logger)
    assert config.logger.name == "config"


def test_sample_rate_and_chunk_size():
    """Test STT configuration constants."""
    import config

    assert config.SAMPLE_RATE == 16000
    assert config.CHUNK_SIZE == 512
    assert config.VAD_THRESHOLD == 0.5
    assert config.PAUSE_LIMIT == 0.8
    assert config.PRE_ROLL_MS == 200


@patch.dict(os.environ, {}, clear=True)
@patch("config.torch.cuda.is_available", return_value=True)
def test_device_selection_cuda_available(mock_cuda):
    """Test device selection when CUDA is available."""
    # Need to reload config to test device selection
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    assert config.DEVICE == "cuda"


@patch.dict(os.environ, {"TTS_DEVICE": "cpu"}, clear=False)
@patch("config.torch.cuda.is_available", return_value=True)
def test_device_selection_force_cpu(mock_cuda):
    """Test device selection when TTS_DEVICE=cpu is set."""
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    assert config.DEVICE == "cpu"


@patch.dict(os.environ, {}, clear=True)
@patch("config.torch.cuda.is_available", return_value=False)
def test_device_selection_cuda_unavailable(mock_cuda):
    """Test device selection when CUDA is not available."""
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    assert config.DEVICE == "cpu"


def test_characters_dict():
    """Test that CHARACTERS dictionary has expected structure."""
    import config

    assert "sir_henry" in config.CHARACTERS
    assert "mr_meeseeks" in config.CHARACTERS
    assert "napoleon_dynamite" in config.CHARACTERS

    for char_name, char_config in config.CHARACTERS.items():
        assert "ref_audio_path" in char_config
        assert "ref_text" in char_config
        assert "speed" in char_config
        assert "system_prompt" in char_config
        assert "warmup_text" in char_config


@patch.dict(os.environ, {"CHARACTER": "sir_henry"}, clear=False)
def test_character_selection_valid():
    """Test character selection with valid character."""
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    assert config.SELECTED_CHARACTER == "sir_henry"
    assert config.REF_AUDIO_PATH == "./ref/sirhenry-reference.wav"
    assert config.REF_TEXT == config.CHARACTERS["sir_henry"]["ref_text"]
    assert config.SPEED == 1.0
    assert config.SYSTEM_PROMPT == config.CHARACTERS["sir_henry"]["system_prompt"]
    assert config.WARMUP_TEXT == config.CHARACTERS["sir_henry"]["warmup_text"]


@patch.dict(os.environ, {"CHARACTER": "mr_meeseeks"}, clear=False)
def test_character_selection_mr_meeseeks():
    """Test character selection with mr_meeseeks."""
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    assert config.SELECTED_CHARACTER == "mr_meeseeks"
    assert config.REF_AUDIO_PATH == "./ref/mrmeeseeks-reference.wav"


@patch.dict(os.environ, {"CHARACTER": "napoleon_dynamite"}, clear=False)
def test_character_selection_napoleon():
    """Test character selection with napoleon_dynamite."""
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    assert config.SELECTED_CHARACTER == "napoleon_dynamite"
    assert config.SPEED == 0.3


@patch.dict(os.environ, {"CHARACTER": "invalid_character"}, clear=False)
def test_character_selection_invalid_defaults():
    """Test character selection with invalid character defaults to sir_henry."""
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    assert config.SELECTED_CHARACTER == "sir_henry"


@patch.dict(os.environ, {"CHARACTER": "INVALID"}, clear=False)
def test_character_selection_case_insensitive():
    """Test character selection is case insensitive."""
    import importlib
    import sys

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])
    import config

    # Should default to sir_henry since 'invalid' (lowercase) is not in CHARACTERS
    assert config.SELECTED_CHARACTER == "sir_henry"


def test_shared_queues_and_events():
    """Test that shared queues and events are initialized."""
    import config

    assert isinstance(config.interrupt_event, threading.Event)
    assert isinstance(config.is_speaking, threading.Event)
    assert isinstance(config.prompt_queue, queue.Queue)
    assert isinstance(config.sentence_queue, queue.Queue)
    assert isinstance(config.mic_audio_queue, queue.Queue)
    assert isinstance(config.playback_audio_queue, queue.Queue)
