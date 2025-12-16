"""Unit tests for config.py module."""

import importlib
import logging
import os
import sys
from unittest.mock import patch, MagicMock


class TestLogging:
    """Test logging configuration."""

    def test_logger_exists(self):
        """Test that logger is configured."""
        import config

        assert isinstance(config.logger, logging.Logger)
        assert config.logger.name == "sir_henry"


class TestLiveKitConfig:
    """Test LiveKit configuration."""

    def test_default_livekit_url(self):
        """Test default LiveKit URL."""
        with patch.dict(os.environ, {}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.LIVEKIT_URL == "ws://localhost:7880"

    def test_custom_livekit_url(self):
        """Test custom LiveKit URL from environment."""
        with patch.dict(os.environ, {"LIVEKIT_URL": "ws://custom:9999"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.LIVEKIT_URL == "ws://custom:9999"

    def test_default_api_key_and_secret(self):
        """Test default API key and secret."""
        with patch.dict(os.environ, {}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.LIVEKIT_API_KEY == "devkey"
            assert config.LIVEKIT_API_SECRET == "secret"


class TestDeviceConfig:
    """Test device configuration."""

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_available(self, mock_cuda):
        """Test device selection when CUDA is available."""
        with patch.dict(os.environ, {}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.DEVICE == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_cuda_unavailable(self, mock_cuda):
        """Test device selection when CUDA is not available."""
        with patch.dict(os.environ, {}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.DEVICE == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_force_cpu(self, mock_cuda):
        """Test device selection when TTS_DEVICE=cpu is set."""
        with patch.dict(os.environ, {"TTS_DEVICE": "cpu"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.DEVICE == "cpu"

    def test_stt_device_default(self):
        """Test STT device defaults to cpu."""
        with patch.dict(os.environ, {}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.STT_DEVICE == "cpu"

    def test_stt_device_custom(self):
        """Test STT device from environment."""
        with patch.dict(os.environ, {"STT_DEVICE": "CUDA"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.STT_DEVICE == "cuda"


class TestOllamaConfig:
    """Test Ollama configuration."""

    def test_default_ollama_host(self):
        """Test default Ollama host."""
        with patch.dict(os.environ, {}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.OLLAMA_HOST == "localhost:11434"

    def test_custom_ollama_host(self):
        """Test custom Ollama host."""
        with patch.dict(os.environ, {"OLLAMA_HOST": "ollama:11434"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.OLLAMA_HOST == "ollama:11434"

    def test_default_ollama_model(self):
        """Test default Ollama model."""
        with patch.dict(os.environ, {}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.OLLAMA_MODEL == "llama3.2:3b"


class TestCharacterConfig:
    """Test character configuration."""

    def test_characters_dict_structure(self):
        """Test CHARACTERS dictionary has expected structure."""
        import config

        assert "sir_henry" in config.CHARACTERS
        assert "mr_meeseeks" in config.CHARACTERS
        assert "napoleon_dynamite" in config.CHARACTERS

        for char_name, char_config in config.CHARACTERS.items():
            assert "ref_audio_path" in char_config
            assert "ref_text" in char_config
            assert "speed" in char_config
            assert "system_prompt" in char_config
            assert "greeting" in char_config

    def test_character_selection_sir_henry(self):
        """Test character selection with sir_henry."""
        with patch.dict(os.environ, {"CHARACTER": "sir_henry"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SELECTED_CHARACTER == "sir_henry"
            assert config.REF_AUDIO_PATH == "./ref/sirhenry-reference.wav"
            assert config.SPEED == 1.0

    def test_character_selection_mr_meeseeks(self):
        """Test character selection with mr_meeseeks."""
        with patch.dict(os.environ, {"CHARACTER": "mr_meeseeks"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SELECTED_CHARACTER == "mr_meeseeks"
            assert config.REF_AUDIO_PATH == "./ref/mrmeeseeks-reference.wav"

    def test_character_selection_napoleon_dynamite(self):
        """Test character selection with napoleon_dynamite."""
        with patch.dict(os.environ, {"CHARACTER": "napoleon_dynamite"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SELECTED_CHARACTER == "napoleon_dynamite"
            assert config.SPEED == 0.3

    def test_character_selection_invalid_defaults(self):
        """Test character selection with invalid character defaults to sir_henry."""
        with patch.dict(os.environ, {"CHARACTER": "invalid_character"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SELECTED_CHARACTER == "sir_henry"

    def test_character_selection_case_insensitive(self):
        """Test character selection converts to lowercase."""
        with patch.dict(os.environ, {"CHARACTER": "MR_MEESEEKS"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.SELECTED_CHARACTER == "mr_meeseeks"

    def test_character_exports(self):
        """Test that character config exports are correct."""
        with patch.dict(os.environ, {"CHARACTER": "sir_henry"}, clear=True):
            if "config" in sys.modules:
                del sys.modules["config"]
            import config

            assert config.REF_TEXT == config.CHARACTERS["sir_henry"]["ref_text"]
            assert (
                config.SYSTEM_PROMPT == config.CHARACTERS["sir_henry"]["system_prompt"]
            )
            assert config.GREETING == config.CHARACTERS["sir_henry"]["greeting"]
