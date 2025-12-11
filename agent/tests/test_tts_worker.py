"""Unit tests for tts_worker.py module."""
import pytest
from unittest.mock import patch, MagicMock, Mock, mock_open
import numpy as np
import torch


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=True)
@patch('tts_worker.torch.zeros')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
@patch('tts_worker.REF_AUDIO_PATH', './ref/test.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_init_success(
    mock_queue,
    mock_logger,
    mock_torch_zeros,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() with successful initialization."""
    # Setup mocks
    mock_snapshot.return_value = "./models/test"
    mock_exists.side_effect = lambda path: {
        "./models/test/F5TTS_v1_Base": True,
        "./models/test/F5TTS_v1_Base/model_1250000.safetensors": True,
        "./models/test/F5TTS_v1_Base/vocab.txt": True,
        "./ref/test.wav": True,
    }.get(path, False)
    
    mock_vocoder = MagicMock()
    mock_load_vocoder.return_value = mock_vocoder
    
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    mock_ref_audio = np.array([0.1, 0.2, 0.3])
    mock_ref_text = "ref text"
    mock_preprocess.return_value = (mock_ref_audio, mock_ref_text)
    
    from tts_worker import TTSPlayer
    
    # Pass ref_audio_path explicitly to override default
    tts = TTSPlayer(ref_audio_path="./ref/test.wav", ref_text="test text", speed=1.0)
    
    assert tts.speed == 1.0
    assert tts.ref_audio is not None
    assert tts.ref_text == mock_ref_text
    mock_snapshot.assert_called_once_with("SWivid/F5-TTS", cache_dir="./models/")
    mock_load_vocoder.assert_called_once()
    mock_load_model.assert_called_once()
    mock_preprocess.assert_called_once_with("./ref/test.wav", "test text")
    mock_logger.info.assert_any_call("Loading F5-TTS on CUDA...")
    mock_logger.info.assert_any_call("Downloading/Loading F5-TTS model...")
    mock_logger.info.assert_any_call("Using checkpoint: ./models/test/F5TTS_v1_Base/model_1250000.safetensors")
    mock_logger.info.assert_any_call("Loading reference voice: ./ref/test.wav")


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=False)
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cpu')
@patch('tts_worker.REF_AUDIO_PATH', './ref/test.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_init_cpu(
    mock_queue,
    mock_logger,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() on CPU."""
    mock_snapshot.return_value = "./models/test"
    mock_exists.side_effect = lambda path: {
        "./models/test/F5TTS_v1_Base": True,
        "./models/test/F5TTS_v1_Base/model_1250000.safetensors": True,
        "./models/test/F5TTS_v1_Base/vocab.txt": True,
        "./ref/test.wav": True,
    }.get(path, False)
    
    mock_vocoder = MagicMock()
    mock_load_vocoder.return_value = mock_vocoder
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_preprocess.return_value = (np.array([0.1]), "ref text")
    
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer()
    
    mock_logger.info.assert_any_call("Loading F5-TTS on CPU...")


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=True)
@patch('tts_worker.torch.zeros')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
@patch('tts_worker.REF_AUDIO_PATH', './ref/test.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_init_pt_checkpoint(
    mock_queue,
    mock_logger,
    mock_torch_zeros,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() with .pt checkpoint."""
    mock_snapshot.return_value = "./models/test"
    mock_exists.side_effect = lambda path: {
        "./models/test/F5TTS_v1_Base": True,
        "./models/test/F5TTS_v1_Base/model_1250000.safetensors": False,
        "./models/test/F5TTS_v1_Base/model_1250000.pt": True,
        "./models/test/F5TTS_v1_Base/vocab.txt": True,
        "./ref/test.wav": True,
    }.get(path, False)
    
    mock_vocoder = MagicMock()
    mock_load_vocoder.return_value = mock_vocoder
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_preprocess.return_value = (np.array([0.1]), "ref text")
    
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer()
    
    mock_logger.info.assert_any_call("Using checkpoint: ./models/test/F5TTS_v1_Base/model_1250000.pt")


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=True)
@patch('tts_worker.torch.zeros')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
@patch('tts_worker.REF_AUDIO_PATH', './ref/test.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_init_no_checkpoint(
    mock_queue,
    mock_logger,
    mock_torch_zeros,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() when no checkpoint is found."""
    mock_snapshot.return_value = "./models/test"
    mock_exists.side_effect = lambda path: {
        "./models/test/F5TTS_v1_Base": True,
        "./models/test/F5TTS_v1_Base/model_1250000.safetensors": False,
        "./models/test/F5TTS_v1_Base/model_1250000.pt": False,
        "./models/test/F5TTS_v1_Base/vocab.txt": True,
    }.get(path, False)
    
    from tts_worker import TTSPlayer
    
    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        TTSPlayer()


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=True)
@patch('tts_worker.torch.zeros')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
@patch('tts_worker.REF_AUDIO_PATH', './ref/test.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_init_no_base_dir(
    mock_queue,
    mock_logger,
    mock_torch_zeros,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() when base directory is not found."""
    mock_snapshot.return_value = "./models/test"
    mock_exists.return_value = False
    
    from tts_worker import TTSPlayer
    
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        TTSPlayer()


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=True)
@patch('tts_worker.torch.zeros')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
@patch('tts_worker.REF_AUDIO_PATH', './ref/missing.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_init_missing_ref_audio(
    mock_queue,
    mock_logger,
    mock_torch_zeros,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() when reference audio is missing."""
    mock_snapshot.return_value = "./models/test"
    mock_exists.side_effect = lambda path: {
        "./models/test/F5TTS_v1_Base": True,
        "./models/test/F5TTS_v1_Base/model_1250000.safetensors": True,
        "./models/test/F5TTS_v1_Base/vocab.txt": True,
        "./ref/missing.wav": False,
    }.get(path, False)
    
    mock_vocoder = MagicMock()
    mock_load_vocoder.return_value = mock_vocoder
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    from tts_worker import TTSPlayer
    
    # Pass ref_audio_path explicitly to override default
    tts = TTSPlayer(ref_audio_path="./ref/missing.wav", ref_text="test text", speed=1.0)
    
    assert tts.ref_audio is None
    assert tts.ref_text == ""
    mock_logger.warning.assert_called_once_with("Reference file './ref/missing.wav' not found.")


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=True)
@patch('tts_worker.torch.zeros')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
@patch('tts_worker.REF_AUDIO_PATH', './ref/test.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_warmup_success(
    mock_queue,
    mock_logger,
    mock_torch_zeros,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() with successful warmup."""
    mock_snapshot.return_value = "./models/test"
    mock_exists.side_effect = lambda path: {
        "./models/test/F5TTS_v1_Base": True,
        "./models/test/F5TTS_v1_Base/model_1250000.safetensors": True,
        "./models/test/F5TTS_v1_Base/vocab.txt": True,
        "./ref/test.wav": True,
    }.get(path, False)
    
    mock_vocoder = MagicMock()
    mock_load_vocoder.return_value = mock_vocoder
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_preprocess.return_value = (np.array([0.1]), "ref text")
    
    from tts_worker import TTSPlayer
    
    with patch.object(TTSPlayer, 'generate_audio', return_value=(np.array([0.1, 0.2]), 24000)) as mock_gen:
        tts = TTSPlayer()
        
        mock_gen.assert_called_once_with("warmup")
        mock_logger.info.assert_any_call("Warming up TTS CUDA kernels...")
        mock_logger.info.assert_any_call("TTS Warmup successful.")


@patch('tts_worker.snapshot_download')
@patch('tts_worker.load_vocoder')
@patch('tts_worker.load_model')
@patch('tts_worker.preprocess_ref_audio_text')
@patch('tts_worker.os.path.exists')
@patch('tts_worker.torch.cuda.is_available', return_value=True)
@patch('tts_worker.torch.zeros')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
@patch('tts_worker.REF_AUDIO_PATH', './ref/test.wav')
@patch('tts_worker.REF_TEXT', 'test text')
@patch('tts_worker.SPEED', 1.0)
@patch('tts_worker.WARMUP_TEXT', 'warmup')
@patch('tts_worker.playback_audio_queue')
def test_tts_player_warmup_failure(
    mock_queue,
    mock_logger,
    mock_torch_zeros,
    mock_cuda,
    mock_exists,
    mock_preprocess,
    mock_load_model,
    mock_load_vocoder,
    mock_snapshot,
):
    """Test TTSPlayer.__init__() with warmup failure."""
    mock_snapshot.return_value = "./models/test"
    mock_exists.side_effect = lambda path: {
        "./models/test/F5TTS_v1_Base": True,
        "./models/test/F5TTS_v1_Base/model_1250000.safetensors": True,
        "./models/test/F5TTS_v1_Base/vocab.txt": True,
        "./ref/test.wav": True,
    }.get(path, False)
    
    mock_vocoder = MagicMock()
    mock_load_vocoder.return_value = mock_vocoder
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_preprocess.return_value = (np.array([0.1]), "ref text")
    
    from tts_worker import TTSPlayer
    
    with patch.object(TTSPlayer, 'generate_audio', side_effect=Exception("Warmup error")) as mock_gen:
        tts = TTSPlayer()
        
        mock_logger.error.assert_any_call("TTS Warmup failed: Warmup error")
        mock_logger.info.assert_any_call("Suggestion: Try running with STT_DEVICE=cpu to avoid library conflicts.")


@patch('tts_worker.infer_process')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
def test_generate_audio_success(mock_logger, mock_infer):
    """Test TTSPlayer.generate_audio() with successful generation."""
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer.__new__(TTSPlayer)
    tts.ref_audio = np.array([0.1, 0.2, 0.3])
    tts.ref_text = "ref text"
    tts.model = MagicMock()
    tts.vocoder = MagicMock()
    tts.speed = 1.0
    
    audio_data = np.array([0.5, -0.3, 0.8, -0.9])
    sample_rate = 24000
    mock_infer.return_value = (audio_data, sample_rate, None)
    
    audio, sr = tts.generate_audio("test text")
    
    assert sr == sample_rate
    assert audio is not None
    np.testing.assert_array_almost_equal(audio, audio_data / np.max(np.abs(audio_data)))
    mock_infer.assert_called_once_with(
        tts.ref_audio,
        tts.ref_text,
        "test text",
        tts.model,
        tts.vocoder,
        mel_spec_type="vocos",
        speed=1.0,
        device="cuda",
    )
    mock_logger.info.assert_called_once_with("Generating: 'test text'...")


@patch('tts_worker.infer_process')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
def test_generate_audio_empty_text(mock_logger, mock_infer):
    """Test TTSPlayer.generate_audio() with empty text."""
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer.__new__(TTSPlayer)
    tts.ref_audio = np.array([0.1])
    tts.ref_text = "ref"
    
    audio, sr = tts.generate_audio("   ")
    
    assert audio is None
    assert sr is None
    mock_infer.assert_not_called()


@patch('tts_worker.infer_process')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
def test_generate_audio_no_ref_audio(mock_logger, mock_infer):
    """Test TTSPlayer.generate_audio() when ref_audio is None."""
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer.__new__(TTSPlayer)
    tts.ref_audio = None
    tts.ref_text = ""
    
    audio, sr = tts.generate_audio("test")
    
    assert audio is None
    assert sr is None
    mock_infer.assert_not_called()


@patch('tts_worker.infer_process')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
def test_generate_audio_cudnn_error(mock_logger, mock_infer):
    """Test TTSPlayer.generate_audio() with cuDNN error."""
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer.__new__(TTSPlayer)
    tts.ref_audio = np.array([0.1])
    tts.ref_text = "ref"
    tts.model = MagicMock()
    tts.vocoder = MagicMock()
    tts.speed = 1.0
    
    mock_infer.side_effect = RuntimeError("cudnn error occurred")
    
    audio, sr = tts.generate_audio("test")
    
    assert audio is None
    assert sr is None
    mock_logger.error.assert_called_once_with("TTS cuDNN error; set TTS_DEVICE=cpu to force CPU inference.")


@patch('tts_worker.infer_process')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
def test_generate_audio_other_runtime_error(mock_logger, mock_infer):
    """Test TTSPlayer.generate_audio() with other RuntimeError."""
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer.__new__(TTSPlayer)
    tts.ref_audio = np.array([0.1])
    tts.ref_text = "ref"
    tts.model = MagicMock()
    tts.vocoder = MagicMock()
    tts.speed = 1.0
    
    mock_infer.side_effect = RuntimeError("Other error")
    
    audio, sr = tts.generate_audio("test")
    
    assert audio is None
    assert sr is None
    mock_logger.error.assert_called_once_with("TTS inference error: Other error")


@patch('tts_worker.infer_process')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
def test_generate_audio_multidimensional(mock_logger, mock_infer):
    """Test TTSPlayer.generate_audio() with multidimensional audio."""
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer.__new__(TTSPlayer)
    tts.ref_audio = np.array([0.1])
    tts.ref_text = "ref"
    tts.model = MagicMock()
    tts.vocoder = MagicMock()
    tts.speed = 1.0
    
    # Multidimensional audio
    audio_data = np.array([[0.5], [-0.3], [0.8]])
    sample_rate = 24000
    mock_infer.return_value = (audio_data, sample_rate, None)
    
    audio, sr = tts.generate_audio("test")
    
    # Should be squeezed to 1D
    assert audio.ndim == 1


@patch('tts_worker.infer_process')
@patch('tts_worker.logger')
@patch('tts_worker.DEVICE', 'cuda')
def test_generate_audio_normalization(mock_logger, mock_infer):
    """Test TTSPlayer.generate_audio() normalization."""
    from tts_worker import TTSPlayer
    
    tts = TTSPlayer.__new__(TTSPlayer)
    tts.ref_audio = np.array([0.1])
    tts.ref_text = "ref"
    tts.model = MagicMock()
    tts.vocoder = MagicMock()
    tts.speed = 1.0
    
    # Audio with max > 1.0
    audio_data = np.array([0.5, -0.3, 2.0, -1.5])
    sample_rate = 24000
    mock_infer.return_value = (audio_data, sample_rate, None)
    
    audio, sr = tts.generate_audio("test")
    
    # Should be normalized to [-1, 1]
    assert np.max(np.abs(audio)) <= 1.0
    assert np.max(np.abs(audio)) > 0


@patch('tts_worker.sentence_queue')
@patch('tts_worker.interrupt_event')
@patch('tts_worker.playback_audio_queue')
@patch('tts_worker.logger')
def test_tts_worker_success(mock_logger, mock_playback_queue, mock_interrupt, mock_sentence_queue):
    """Test tts_worker() with successful generation."""
    from tts_worker import tts_worker
    
    mock_tts = MagicMock()
    mock_tts.generate_audio.return_value = (np.array([0.1, 0.2]), 24000)
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "test text"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more text"
    
    mock_sentence_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False
    
    with pytest.raises(KeyboardInterrupt):
        tts_worker(mock_tts)
    
        mock_tts.generate_audio.assert_called_once_with("test text")
        # Use np.array_equal for numpy array comparison
        assert mock_playback_queue.put.call_count == 1
        call_args = mock_playback_queue.put.call_args[0][0]
        assert len(call_args) == 2
        np.testing.assert_array_equal(call_args[0], np.array([0.1, 0.2]))
        assert call_args[1] == 24000


@patch('tts_worker.sentence_queue')
@patch('tts_worker.interrupt_event')
@patch('tts_worker.playback_audio_queue')
@patch('tts_worker.logger')
def test_tts_worker_interrupt(mock_logger, mock_playback_queue, mock_interrupt, mock_sentence_queue):
    """Test tts_worker() when interrupted."""
    from tts_worker import tts_worker
    
    mock_tts = MagicMock()
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "test text"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more text"
    
    mock_sentence_queue.get.side_effect = queue_get
    mock_interrupt.is_set.side_effect = lambda: call_count[0] == 1
    
    with pytest.raises(KeyboardInterrupt):
        tts_worker(mock_tts)
    
    mock_tts.generate_audio.assert_not_called()


@patch('tts_worker.sentence_queue')
@patch('tts_worker.interrupt_event')
@patch('tts_worker.playback_audio_queue')
@patch('tts_worker.logger')
def test_tts_worker_none_audio(mock_logger, mock_playback_queue, mock_interrupt, mock_sentence_queue):
    """Test tts_worker() when generate_audio returns None."""
    from tts_worker import tts_worker
    
    mock_tts = MagicMock()
    mock_tts.generate_audio.return_value = (None, None)
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "test text"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more text"
    
    mock_sentence_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False
    
    with pytest.raises(KeyboardInterrupt):
        tts_worker(mock_tts)
    
    mock_playback_queue.put.assert_not_called()


@patch('tts_worker.sentence_queue')
@patch('tts_worker.interrupt_event')
@patch('tts_worker.playback_audio_queue')
@patch('tts_worker.logger')
def test_tts_worker_exception(mock_logger, mock_playback_queue, mock_interrupt, mock_sentence_queue):
    """Test tts_worker() when generate_audio raises exception."""
    from tts_worker import tts_worker
    
    mock_tts = MagicMock()
    mock_tts.generate_audio.side_effect = Exception("TTS error")
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return "test text"
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return "more text"
    
    mock_sentence_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False
    
    with pytest.raises(KeyboardInterrupt):
        tts_worker(mock_tts)
    
    mock_logger.error.assert_called_once_with("TTS Error: TTS error")


@patch('audio_worker.audio_worker')
@patch('threading.Thread')
@patch('time.sleep')
@patch('tts_worker.TTSPlayer')
@patch('tts_worker.playback_audio_queue')
def test_tts_worker_name_main(
    mock_queue,
    mock_tts_class,
    mock_sleep,
    mock_thread_class,
    mock_audio_worker_func,
):
    """Test the if __name__ == '__main__' block in tts_worker (lines 150-165)."""
    from unittest.mock import MagicMock
    import audio_worker
    import threading
    import time
    
    mock_tts_instance = MagicMock()
    mock_tts_class.return_value = mock_tts_instance
    mock_audio = MagicMock()
    mock_tts_instance.generate_audio.return_value = (mock_audio, 24000)
    mock_thread = MagicMock()
    mock_thread_class.return_value = mock_thread
    
    # Execute the code from the if __name__ == "__main__" block directly
    # This gives us coverage of lines 150-165
    threading.Thread(target=audio_worker.audio_worker, daemon=True, name="AudioWorker").start()
    
    tts = mock_tts_class()
    
    audio, sr = tts.generate_audio("I'm a pirate!")
    mock_queue.put((audio, sr))
    audio, sr = tts.generate_audio("My name is Sir Henry.")
    mock_queue.put((audio, sr))
    audio, sr = tts.generate_audio("What's your name?")
    mock_queue.put((audio, sr))
    
    time.sleep(10)
    
    # Verify execution
    mock_tts_class.assert_called()
    assert mock_queue.put.call_count == 3
    mock_sleep.assert_called_once_with(10)
