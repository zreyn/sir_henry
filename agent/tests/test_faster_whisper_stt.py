"""Unit tests for plugins/faster_whisper_stt.py module."""

import sys
from unittest.mock import MagicMock, patch

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def mock_stt_dependencies():
    """Mock all faster-whisper dependencies."""
    # Mock livekit.agents.stt
    mock_stt = MagicMock()

    class MockSTT:
        def __init__(self, **kwargs):
            pass

    mock_stt.STT = MockSTT
    mock_stt.STTCapabilities = MagicMock
    mock_stt.SpeechEvent = MagicMock
    mock_stt.SpeechEventType = MagicMock()
    mock_stt.SpeechEventType.FINAL_TRANSCRIPT = "final"
    mock_stt.SpeechData = MagicMock
    mock_stt.STTConnOptions = MagicMock

    mock_utils = MagicMock()
    mock_utils.AudioBuffer = MagicMock

    mock_agents = MagicMock()
    mock_agents.stt = mock_stt
    mock_agents.utils = mock_utils

    # Mock faster_whisper
    mock_faster_whisper = MagicMock()
    mock_whisper_model = MagicMock()
    mock_faster_whisper.WhisperModel = MagicMock(return_value=mock_whisper_model)

    with patch.dict(
        "sys.modules",
        {
            "numpy": np,
            "livekit": MagicMock(),
            "livekit.agents": mock_agents,
            "livekit.agents.stt": mock_stt,
            "livekit.agents.utils": mock_utils,
            "faster_whisper": mock_faster_whisper,
        },
    ):
        # Clear cached module
        if "plugins.faster_whisper_stt" in sys.modules:
            del sys.modules["plugins.faster_whisper_stt"]

        yield {
            "stt": mock_stt,
            "utils": mock_utils,
            "faster_whisper": mock_faster_whisper,
            "whisper_model": mock_whisper_model,
        }


def _import_stt_module():
    """Helper to import the STT module directly, bypassing plugins/__init__.py."""
    import importlib.util
    import os

    module_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "plugins", "faster_whisper_stt.py"
    )
    spec = importlib.util.spec_from_file_location("faster_whisper_stt", module_path)
    module = importlib.util.module_from_spec(spec)
    # Register the module in sys.modules BEFORE exec_module so dataclass can find it
    sys.modules["faster_whisper_stt"] = module
    spec.loader.exec_module(module)
    return module


class TestFasterWhisperOptions:
    """Test FasterWhisperOptions dataclass."""

    def test_default_options(self, mock_stt_dependencies):
        """Test default option values."""
        module = _import_stt_module()

        opts = module.FasterWhisperOptions()

        assert opts.model_size == "small"
        assert opts.device == "cpu"
        assert opts.compute_type is None
        assert opts.language is None
        assert opts.beam_size == 1
        assert opts.best_of == 1
        assert opts.vad_filter is True

    def test_custom_options(self, mock_stt_dependencies):
        """Test custom option values."""
        module = _import_stt_module()

        opts = module.FasterWhisperOptions(
            model_size="large",
            device="cuda",
            compute_type="float16",
            language="en",
            beam_size=5,
            best_of=3,
            vad_filter=False,
        )

        assert opts.model_size == "large"
        assert opts.device == "cuda"
        assert opts.compute_type == "float16"
        assert opts.language == "en"
        assert opts.beam_size == 5
        assert opts.best_of == 3
        assert opts.vad_filter is False


class TestFasterWhisperSTT:
    """Test FasterWhisperSTT class."""

    def test_init_default_cpu(self, mock_stt_dependencies):
        """Test initialization with default CPU settings."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        assert stt._opts.model_size == "small"
        assert stt._opts.device == "cpu"
        assert stt._opts.compute_type == "int8"
        assert stt._loaded is False

    def test_init_cuda_device(self, mock_stt_dependencies):
        """Test initialization with CUDA device."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT(device="cuda")

        assert stt._opts.device == "cuda"
        assert stt._opts.compute_type == "float16"

    def test_init_explicit_compute_type(self, mock_stt_dependencies):
        """Test initialization with explicit compute type."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT(device="cuda", compute_type="int8")

        assert stt._opts.compute_type == "int8"

    def test_init_custom_options(self, mock_stt_dependencies):
        """Test initialization with custom options."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT(
            model_size="large",
            device="cpu",
            language="en",
            beam_size=5,
            best_of=3,
            vad_filter=False,
        )

        assert stt._opts.model_size == "large"
        assert stt._opts.language == "en"
        assert stt._opts.beam_size == 5
        assert stt._opts.best_of == 3
        assert stt._opts.vad_filter is False

    def test_ensure_loaded(self, mock_stt_dependencies):
        """Test _ensure_loaded loads the model."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        stt._ensure_loaded()

        assert stt._loaded is True
        mock_stt_dependencies["faster_whisper"].WhisperModel.assert_called_once_with(
            "small",
            device="cpu",
            compute_type="int8",
        )

    def test_ensure_loaded_already_loaded(self, mock_stt_dependencies):
        """Test _ensure_loaded returns early if already loaded."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()
        stt._loaded = True
        stt._model = MagicMock()

        stt._ensure_loaded()

        # Should not have called WhisperModel again
        mock_stt_dependencies["faster_whisper"].WhisperModel.assert_not_called()

    @pytest.mark.asyncio
    async def test_recognize_impl(self, mock_stt_dependencies):
        """Test _recognize_impl transcribes audio."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        # Create mock audio buffer
        mock_buffer = MagicMock()
        mock_buffer.data = np.zeros(16000, dtype=np.int16).tobytes()

        # Setup mock transcription result
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_stt_dependencies["whisper_model"].transcribe.return_value = (
            [mock_segment],
            mock_info,
        )
        stt._model = mock_stt_dependencies["whisper_model"]
        stt._loaded = True

        result = await stt._recognize_impl(mock_buffer)

        assert result is not None

    @pytest.mark.asyncio
    async def test_recognize_impl_with_language(self, mock_stt_dependencies):
        """Test _recognize_impl with explicit language."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        mock_buffer = MagicMock()
        mock_buffer.data = np.zeros(16000, dtype=np.int16).tobytes()

        mock_segment = MagicMock()
        mock_segment.text = "Bonjour"

        mock_info = MagicMock()
        mock_info.language = "fr"

        mock_stt_dependencies["whisper_model"].transcribe.return_value = (
            [mock_segment],
            mock_info,
        )
        stt._model = mock_stt_dependencies["whisper_model"]
        stt._loaded = True

        result = await stt._recognize_impl(mock_buffer, language="fr")

        assert result is not None

    @pytest.mark.asyncio
    async def test_recognize_impl_multiple_segments(self, mock_stt_dependencies):
        """Test _recognize_impl with multiple segments."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        mock_buffer = MagicMock()
        mock_buffer.data = np.zeros(16000, dtype=np.int16).tobytes()

        mock_segment1 = MagicMock()
        mock_segment1.text = "Hello "
        mock_segment2 = MagicMock()
        mock_segment2.text = "world"

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_stt_dependencies["whisper_model"].transcribe.return_value = (
            [mock_segment1, mock_segment2],
            mock_info,
        )
        stt._model = mock_stt_dependencies["whisper_model"]
        stt._loaded = True

        result = await stt._recognize_impl(mock_buffer)

        assert result is not None

    @pytest.mark.asyncio
    async def test_recognize_impl_no_language_detected(self, mock_stt_dependencies):
        """Test _recognize_impl when no language is detected."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        mock_buffer = MagicMock()
        mock_buffer.data = np.zeros(16000, dtype=np.int16).tobytes()

        mock_segment = MagicMock()
        mock_segment.text = "Hello"

        mock_info = MagicMock()
        mock_info.language = None

        mock_stt_dependencies["whisper_model"].transcribe.return_value = (
            [mock_segment],
            mock_info,
        )
        stt._model = mock_stt_dependencies["whisper_model"]
        stt._loaded = True

        result = await stt._recognize_impl(mock_buffer)

        assert result is not None

    @pytest.mark.asyncio
    async def test_recognize(self, mock_stt_dependencies):
        """Test recognize method calls _recognize_impl."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        mock_buffer = MagicMock()
        mock_buffer.data = np.zeros(16000, dtype=np.int16).tobytes()

        mock_segment = MagicMock()
        mock_segment.text = "Hello"

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_stt_dependencies["whisper_model"].transcribe.return_value = (
            [mock_segment],
            mock_info,
        )
        stt._model = mock_stt_dependencies["whisper_model"]
        stt._loaded = True

        result = await stt.recognize(mock_buffer)

        assert result is not None

    @pytest.mark.asyncio
    async def test_recognize_with_conn_options(self, mock_stt_dependencies):
        """Test recognize with connection options."""
        module = _import_stt_module()

        stt = module.FasterWhisperSTT()

        mock_buffer = MagicMock()
        mock_buffer.data = np.zeros(16000, dtype=np.int16).tobytes()

        mock_segment = MagicMock()
        mock_segment.text = "Hello"

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_stt_dependencies["whisper_model"].transcribe.return_value = (
            [mock_segment],
            mock_info,
        )
        stt._model = mock_stt_dependencies["whisper_model"]
        stt._loaded = True

        mock_conn_opts = MagicMock()
        result = await stt.recognize(mock_buffer, conn_options=mock_conn_opts)

        assert result is not None
