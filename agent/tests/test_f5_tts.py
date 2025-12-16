"""Unit tests for plugins/f5_tts.py module."""

import asyncio
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def mock_f5_dependencies():
    """Mock all F5-TTS dependencies."""
    # Create mock modules
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.zeros.return_value = MagicMock()

    mock_snapshot_download = MagicMock(return_value="/mock/model/dir")

    mock_dit = MagicMock()

    mock_utils_infer = MagicMock()
    mock_utils_infer.load_model = MagicMock(return_value=MagicMock())
    mock_utils_infer.load_vocoder = MagicMock(return_value=MagicMock())
    mock_utils_infer.infer_process = MagicMock(
        return_value=(np.zeros(1000, dtype=np.float32), 24000, None)
    )
    mock_utils_infer.preprocess_ref_audio_text = MagicMock(
        return_value=("mock_audio", "mock_text")
    )

    # Mock livekit.agents.tts
    mock_tts = MagicMock()
    mock_tts.TTS = type(
        "TTS",
        (),
        {
            "__init__": lambda self, **kwargs: None,
            "sample_rate": 24000,
            "num_channels": 1,
        },
    )
    mock_tts.TTSCapabilities = MagicMock
    mock_tts.ChunkedStream = type(
        "ChunkedStream",
        (),
        {
            "__init__": lambda self, **kwargs: setattr(
                self, "_input_text", kwargs.get("input_text", "")
            ),
        },
    )

    mock_api_connect_options = MagicMock

    mock_agents = MagicMock()
    mock_agents.tts = mock_tts
    mock_agents.APIConnectOptions = mock_api_connect_options

    with patch.dict(
        "sys.modules",
        {
            "torch": mock_torch,
            "numpy": np,
            "huggingface_hub": MagicMock(snapshot_download=mock_snapshot_download),
            "f5_tts": MagicMock(),
            "f5_tts.model": MagicMock(DiT=mock_dit),
            "f5_tts.infer": MagicMock(),
            "f5_tts.infer.utils_infer": mock_utils_infer,
            "livekit": MagicMock(),
            "livekit.agents": mock_agents,
            "livekit.agents.tts": mock_tts,
            "livekit.agents.tts.tts": MagicMock(),
        },
    ):
        # Clear cached module
        if "plugins.f5_tts" in sys.modules:
            del sys.modules["plugins.f5_tts"]

        yield {
            "torch": mock_torch,
            "utils_infer": mock_utils_infer,
            "snapshot_download": mock_snapshot_download,
            "tts": mock_tts,
            "agents": mock_agents,
        }


class TestF5TTS:
    """Test F5TTS class."""

    def test_init_default_device_cpu(self, mock_f5_dependencies):
        """Test F5TTS initialization with default CPU device."""
        mock_f5_dependencies["torch"].cuda.is_available.return_value = False

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        assert tts._device == "cpu"
        assert tts.speed == 1.0
        assert tts._ref_audio_path == "/path/to/ref.wav"
        assert tts._ref_text_input == "Reference text"
        assert tts._loaded is False

    def test_init_default_device_cuda(self, mock_f5_dependencies):
        """Test F5TTS initialization with CUDA available."""
        mock_f5_dependencies["torch"].cuda.is_available.return_value = True

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        assert tts._device == "cuda"

    def test_init_explicit_device(self, mock_f5_dependencies):
        """Test F5TTS initialization with explicit device."""
        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
            device="cpu",
        )

        assert tts._device == "cpu"

    def test_init_custom_speed(self, mock_f5_dependencies):
        """Test F5TTS initialization with custom speed."""
        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
            speed=1.5,
        )

        assert tts.speed == 1.5

    @patch("os.path.exists")
    def test_ensure_loaded_safetensors(self, mock_exists, mock_f5_dependencies):
        """Test _ensure_loaded with safetensors checkpoint."""

        # Setup exists to return True for safetensors path
        def exists_side_effect(path):
            if "safetensors" in path:
                return True
            if "F5TTS_v1_Base" in path:
                return True
            if "vocab.txt" in path:
                return True
            return path == "/path/to/ref.wav"

        mock_exists.side_effect = exists_side_effect

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        tts._ensure_loaded()

        assert tts._loaded is True
        mock_f5_dependencies["utils_infer"].load_vocoder.assert_called_once()
        mock_f5_dependencies["utils_infer"].load_model.assert_called_once()

    @patch("os.path.exists")
    def test_ensure_loaded_pt_checkpoint(self, mock_exists, mock_f5_dependencies):
        """Test _ensure_loaded with .pt checkpoint."""

        def exists_side_effect(path):
            if "safetensors" in path:
                return False
            if ".pt" in path:
                return True
            if "F5TTS_v1_Base" in path:
                return True
            if "vocab.txt" in path:
                return False
            return path == "/path/to/ref.wav"

        mock_exists.side_effect = exists_side_effect

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        tts._ensure_loaded()

        assert tts._loaded is True

    @patch("os.path.exists")
    def test_ensure_loaded_no_checkpoint_raises(
        self, mock_exists, mock_f5_dependencies
    ):
        """Test _ensure_loaded raises when no checkpoint found."""

        def exists_side_effect(path):
            if "safetensors" in path:
                return False
            if ".pt" in path:
                return False
            if "F5TTS_v1_Base" in path:
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            tts._ensure_loaded()

    @patch("os.path.exists")
    def test_ensure_loaded_no_base_dir_raises(self, mock_exists, mock_f5_dependencies):
        """Test _ensure_loaded raises when base dir not found."""
        mock_exists.return_value = False

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            tts._ensure_loaded()

    @patch("os.path.exists")
    def test_ensure_loaded_no_ref_audio_raises(self, mock_exists, mock_f5_dependencies):
        """Test _ensure_loaded raises when reference audio not found."""

        def exists_side_effect(path):
            if "safetensors" in path:
                return True
            if "F5TTS_v1_Base" in path:
                return True
            if "vocab.txt" in path:
                return True
            # Reference audio doesn't exist
            return False

        mock_exists.side_effect = exists_side_effect

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        with pytest.raises(FileNotFoundError, match="Reference audio not found"):
            tts._ensure_loaded()

    @patch("os.path.exists")
    def test_ensure_loaded_cuda_init(self, mock_exists, mock_f5_dependencies):
        """Test _ensure_loaded initializes CUDA when available."""
        mock_f5_dependencies["torch"].cuda.is_available.return_value = True

        def exists_side_effect(path):
            return True

        mock_exists.side_effect = exists_side_effect

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
            device="cuda",
        )

        tts._ensure_loaded()

        # Should have called torch.zeros(1).cuda()
        mock_f5_dependencies["torch"].zeros.assert_called_once_with(1)

    @patch("os.path.exists")
    def test_ensure_loaded_already_loaded(self, mock_exists, mock_f5_dependencies):
        """Test _ensure_loaded returns early if already loaded."""
        mock_exists.return_value = True

        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        tts._loaded = True

        tts._ensure_loaded()

        # Should not have called load functions again
        mock_f5_dependencies["utils_infer"].load_vocoder.assert_not_called()

    def test_synthesize_returns_chunked_stream(self, mock_f5_dependencies):
        """Test synthesize returns a ChunkedStream."""
        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        stream = tts.synthesize("Hello world")

        assert stream is not None

    def test_synthesize_with_conn_options(self, mock_f5_dependencies):
        """Test synthesize with custom connection options."""
        from plugins.f5_tts import F5TTS

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        mock_conn_opts = MagicMock()
        stream = tts.synthesize("Hello world", conn_options=mock_conn_opts)

        assert stream is not None


class TestF5ChunkedStream:
    """Test _F5ChunkedStream class."""

    @pytest.mark.asyncio
    @patch("os.path.exists", return_value=True)
    async def test_run_synthesizes_and_emits(self, mock_exists, mock_f5_dependencies):
        """Test _run synthesizes audio and emits it."""
        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        # Pre-load to avoid file checks
        tts._loaded = True
        tts._ref_audio = "mock_audio"
        tts._ref_text = "mock_text"
        tts._model = MagicMock()
        tts._vocoder = MagicMock()

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello world",
            conn_options=MagicMock(),
        )

        mock_emitter = MagicMock()

        await stream._run(mock_emitter)

        mock_emitter.initialize.assert_called_once()
        mock_emitter.push.assert_called_once()

    @patch("os.path.exists", return_value=True)
    def test_synthesize_blocking_empty_text(self, mock_exists, mock_f5_dependencies):
        """Test _synthesize_blocking returns None for empty text."""
        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="",
            conn_options=MagicMock(),
        )

        result = stream._synthesize_blocking("   ")

        assert result is None

    @patch("os.path.exists", return_value=True)
    def test_synthesize_blocking_no_ref_audio(self, mock_exists, mock_f5_dependencies):
        """Test _synthesize_blocking returns None when ref audio not loaded."""
        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        tts._loaded = True
        tts._ref_audio = None

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello",
            conn_options=MagicMock(),
        )

        result = stream._synthesize_blocking("Hello")

        assert result is None

    @patch("os.path.exists", return_value=True)
    def test_synthesize_blocking_cudnn_error(self, mock_exists, mock_f5_dependencies):
        """Test _synthesize_blocking handles cuDNN errors."""
        mock_f5_dependencies["utils_infer"].infer_process.side_effect = RuntimeError(
            "cuDNN error occurred"
        )

        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        tts._loaded = True
        tts._ref_audio = "mock_audio"
        tts._ref_text = "mock_text"
        tts._model = MagicMock()
        tts._vocoder = MagicMock()

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello",
            conn_options=MagicMock(),
        )

        with pytest.raises(RuntimeError, match="cuDNN"):
            stream._synthesize_blocking("Hello")

    @patch("os.path.exists", return_value=True)
    def test_synthesize_blocking_other_runtime_error(
        self, mock_exists, mock_f5_dependencies
    ):
        """Test _synthesize_blocking re-raises other RuntimeErrors."""
        mock_f5_dependencies["utils_infer"].infer_process.side_effect = RuntimeError(
            "Some other error"
        )

        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        tts._loaded = True
        tts._ref_audio = "mock_audio"
        tts._ref_text = "mock_text"
        tts._model = MagicMock()
        tts._vocoder = MagicMock()

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello",
            conn_options=MagicMock(),
        )

        with pytest.raises(RuntimeError, match="Some other error"):
            stream._synthesize_blocking("Hello")

    @patch("os.path.exists", return_value=True)
    def test_synthesize_blocking_2d_audio(self, mock_exists, mock_f5_dependencies):
        """Test _synthesize_blocking handles 2D audio arrays."""
        # Return 2D audio array
        mock_f5_dependencies["utils_infer"].infer_process.return_value = (
            np.zeros((1, 1000), dtype=np.float32),
            24000,
            None,
        )

        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        tts._loaded = True
        tts._ref_audio = "mock_audio"
        tts._ref_text = "mock_text"
        tts._model = MagicMock()
        tts._vocoder = MagicMock()

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello",
            conn_options=MagicMock(),
        )

        result = stream._synthesize_blocking("Hello")

        assert result is not None
        assert isinstance(result, bytes)

    @patch("os.path.exists", return_value=True)
    def test_synthesize_blocking_normalizes_audio(
        self, mock_exists, mock_f5_dependencies
    ):
        """Test _synthesize_blocking normalizes audio correctly."""
        # Return audio with max value > 1
        audio = np.array([0.5, 1.0, -0.5], dtype=np.float32)
        mock_f5_dependencies["utils_infer"].infer_process.return_value = (
            audio,
            24000,
            None,
        )

        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        tts._loaded = True
        tts._ref_audio = "mock_audio"
        tts._ref_text = "mock_text"
        tts._model = MagicMock()
        tts._vocoder = MagicMock()

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello",
            conn_options=MagicMock(),
        )

        result = stream._synthesize_blocking("Hello")

        assert result is not None

    @patch("os.path.exists", return_value=True)
    def test_synthesize_blocking_zero_audio(self, mock_exists, mock_f5_dependencies):
        """Test _synthesize_blocking handles zero audio (no normalization needed)."""
        # Return all zeros
        audio = np.zeros(1000, dtype=np.float32)
        mock_f5_dependencies["utils_infer"].infer_process.return_value = (
            audio,
            24000,
            None,
        )

        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        tts._loaded = True
        tts._ref_audio = "mock_audio"
        tts._ref_text = "mock_text"
        tts._model = MagicMock()
        tts._vocoder = MagicMock()

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello",
            conn_options=MagicMock(),
        )

        result = stream._synthesize_blocking("Hello")

        assert result is not None

    @pytest.mark.asyncio
    @patch("os.path.exists", return_value=True)
    async def test_run_no_audio_bytes(self, mock_exists, mock_f5_dependencies):
        """Test _run handles None audio bytes (doesn't push)."""
        from plugins.f5_tts import F5TTS, _F5ChunkedStream

        tts = F5TTS(
            ref_audio_path="/path/to/ref.wav",
            ref_text="Reference text",
        )
        tts._loaded = True
        tts._ref_audio = None  # Will cause None return

        stream = _F5ChunkedStream(
            tts_plugin=tts,
            input_text="Hello",
            conn_options=MagicMock(),
        )

        mock_emitter = MagicMock()

        await stream._run(mock_emitter)

        mock_emitter.initialize.assert_called_once()
        mock_emitter.push.assert_not_called()
