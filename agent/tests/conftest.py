"""Pytest configuration and fixtures."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def mock_torch():
    """Mock torch module."""
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        mock = MagicMock()
        mock.cuda.is_available.return_value = False
        mock.zeros.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_livekit_agents():
    """Mock livekit.agents module."""
    mock_tts = MagicMock()
    mock_tts.TTS = MagicMock
    mock_tts.TTSCapabilities = MagicMock
    mock_tts.ChunkedStream = MagicMock

    mock_stt = MagicMock()
    mock_stt.STT = MagicMock
    mock_stt.STTCapabilities = MagicMock
    mock_stt.SpeechEvent = MagicMock
    mock_stt.SpeechEventType = MagicMock()
    mock_stt.SpeechEventType.FINAL_TRANSCRIPT = "final"
    mock_stt.SpeechData = MagicMock

    mock_agents = MagicMock()
    mock_agents.tts = mock_tts
    mock_agents.stt = mock_stt
    mock_agents.utils = MagicMock()
    mock_agents.APIConnectOptions = MagicMock

    return mock_agents


@pytest.fixture
def mock_f5_tts_deps():
    """Mock F5-TTS dependencies."""
    return {
        "huggingface_hub": MagicMock(),
        "f5_tts.model": MagicMock(),
        "f5_tts.infer.utils_infer": MagicMock(),
    }


@pytest.fixture
def mock_faster_whisper():
    """Mock faster-whisper module."""
    mock = MagicMock()
    mock.WhisperModel = MagicMock()
    return mock
