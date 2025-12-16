"""Unit tests for plugins/__init__.py module."""

import sys
from unittest.mock import MagicMock, patch


class TestPluginsInit:
    """Test plugins __init__.py exports."""

    def test_exports_f5tts(self):
        """Test that F5TTS is exported."""
        # Mock the heavy dependencies
        mock_f5_tts_module = MagicMock()
        mock_f5_tts_module.F5TTS = MagicMock()

        mock_faster_whisper_module = MagicMock()
        mock_faster_whisper_module.FasterWhisperSTT = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "numpy": MagicMock(),
                "huggingface_hub": MagicMock(),
                "f5_tts": MagicMock(),
                "f5_tts.model": MagicMock(),
                "f5_tts.infer": MagicMock(),
                "f5_tts.infer.utils_infer": MagicMock(),
                "livekit": MagicMock(),
                "livekit.agents": MagicMock(),
                "livekit.agents.tts": MagicMock(),
                "livekit.agents.stt": MagicMock(),
                "livekit.agents.utils": MagicMock(),
                "faster_whisper": MagicMock(),
                "plugins.f5_tts": mock_f5_tts_module,
                "plugins.faster_whisper_stt": mock_faster_whisper_module,
            },
        ):
            # Remove cached plugins module
            if "plugins" in sys.modules:
                del sys.modules["plugins"]

            from plugins import F5TTS, FasterWhisperSTT

            assert F5TTS is not None
            assert FasterWhisperSTT is not None

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        mock_f5_tts_module = MagicMock()
        mock_f5_tts_module.F5TTS = MagicMock()

        mock_faster_whisper_module = MagicMock()
        mock_faster_whisper_module.FasterWhisperSTT = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "torch": MagicMock(),
                "numpy": MagicMock(),
                "huggingface_hub": MagicMock(),
                "f5_tts": MagicMock(),
                "f5_tts.model": MagicMock(),
                "f5_tts.infer": MagicMock(),
                "f5_tts.infer.utils_infer": MagicMock(),
                "livekit": MagicMock(),
                "livekit.agents": MagicMock(),
                "livekit.agents.tts": MagicMock(),
                "livekit.agents.stt": MagicMock(),
                "livekit.agents.utils": MagicMock(),
                "faster_whisper": MagicMock(),
                "plugins.f5_tts": mock_f5_tts_module,
                "plugins.faster_whisper_stt": mock_faster_whisper_module,
            },
        ):
            if "plugins" in sys.modules:
                del sys.modules["plugins"]

            import plugins

            assert "F5TTS" in plugins.__all__
            assert "FasterWhisperSTT" in plugins.__all__
