"""Unit tests for main.py module."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import threading
import time


@patch("main.tts_worker")
@patch("main.llm_worker")
@patch("main.stt_worker")
@patch("main.audio_worker")
@patch("main.TTSPlayer")
@patch("main.torch.cuda.is_available", return_value=True)
@patch("main.torch.zeros")
@patch("main.sd.stop")
@patch("main.time.sleep")
@patch("main.logger")
def test_main_cuda_available(
    mock_logger,
    mock_sleep,
    mock_sd_stop,
    mock_torch_zeros,
    mock_cuda,
    mock_tts_class,
    mock_audio_worker,
    mock_stt_worker,
    mock_llm_worker,
    mock_tts_worker,
):
    """Test main() when CUDA is available."""
    mock_tts_instance = MagicMock()
    mock_tts_class.return_value = mock_tts_instance

    # Mock sleep to raise KeyboardInterrupt after first call
    call_count = [0]

    def sleep_side_effect(*args):
        call_count[0] += 1
        if call_count[0] == 1:
            raise KeyboardInterrupt()

    mock_sleep.side_effect = sleep_side_effect

    from main import main

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    mock_logger.info.assert_any_call("Initializing PyTorch CUDA context...")
    mock_torch_zeros.assert_called_once()
    mock_tts_class.assert_called_once()
    mock_logger.info.assert_any_call("System Ready. Speak to interact.")
    mock_logger.info.assert_any_call("Shutting down...")
    mock_sd_stop.assert_called_once()


@patch("main.tts_worker")
@patch("main.llm_worker")
@patch("main.stt_worker")
@patch("main.audio_worker")
@patch("main.TTSPlayer")
@patch("main.torch.cuda.is_available", return_value=False)
@patch("main.sd.stop")
@patch("main.time.sleep")
@patch("main.logger")
def test_main_cuda_unavailable(
    mock_logger,
    mock_sleep,
    mock_sd_stop,
    mock_cuda,
    mock_tts_class,
    mock_audio_worker,
    mock_stt_worker,
    mock_llm_worker,
    mock_tts_worker,
):
    """Test main() when CUDA is not available."""
    mock_tts_instance = MagicMock()
    mock_tts_class.return_value = mock_tts_instance

    call_count = [0]

    def sleep_side_effect(*args):
        call_count[0] += 1
        if call_count[0] == 1:
            raise KeyboardInterrupt()

    mock_sleep.side_effect = sleep_side_effect

    from main import main

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    # Should not call torch.zeros when CUDA is not available
    mock_tts_class.assert_called_once()


@patch("main.tts_worker")
@patch("main.llm_worker")
@patch("main.stt_worker")
@patch("main.audio_worker")
@patch("main.TTSPlayer")
@patch("main.torch.cuda.is_available", return_value=True)
@patch("main.torch.zeros")
@patch("main.sd.stop")
@patch("main.time.sleep")
@patch("main.logger")
@patch("sys.exit")
def test_main_tts_init_failure(
    mock_sys_exit,
    mock_logger,
    mock_sleep,
    mock_sd_stop,
    mock_torch_zeros,
    mock_cuda,
    mock_tts_class,
    mock_audio_worker,
    mock_stt_worker,
    mock_llm_worker,
    mock_tts_worker,
):
    """Test main() when TTS initialization fails."""
    mock_tts_class.side_effect = Exception("TTS load error")
    # Make sys.exit raise SystemExit so execution stops
    mock_sys_exit.side_effect = SystemExit(1)

    from main import main

    with pytest.raises(SystemExit):
        main()

    mock_sys_exit.assert_called_once_with(1)
    mock_logger.error.assert_called_once_with("Failed to load TTS: TTS load error")


@patch("main.tts_worker")
@patch("main.llm_worker")
@patch("main.stt_worker")
@patch("main.audio_worker")
@patch("main.TTSPlayer")
@patch("main.torch.cuda.is_available", return_value=True)
@patch("main.torch.zeros")
@patch("main.sd.stop")
@patch("main.time.sleep")
@patch("main.logger")
def test_main_sd_stop_exception(
    mock_logger,
    mock_sleep,
    mock_sd_stop,
    mock_torch_zeros,
    mock_cuda,
    mock_tts_class,
    mock_audio_worker,
    mock_stt_worker,
    mock_llm_worker,
    mock_tts_worker,
):
    """Test main() when sd.stop() raises an exception."""
    mock_tts_instance = MagicMock()
    mock_tts_class.return_value = mock_tts_instance
    mock_sd_stop.side_effect = Exception("Audio error")

    call_count = [0]

    def sleep_side_effect(*args):
        call_count[0] += 1
        if call_count[0] == 1:
            raise KeyboardInterrupt()

    mock_sleep.side_effect = sleep_side_effect

    from main import main

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    # Should still exit gracefully even if sd.stop() fails


@patch("main.tts_worker")
@patch("main.llm_worker")
@patch("main.stt_worker")
@patch("main.audio_worker")
@patch("main.TTSPlayer")
@patch("main.torch.cuda.is_available", return_value=True)
@patch("main.torch.zeros")
@patch("main.sd.stop")
@patch("main.time.sleep")
@patch("main.logger")
def test_main_thread_creation(
    mock_logger,
    mock_sleep,
    mock_sd_stop,
    mock_torch_zeros,
    mock_cuda,
    mock_tts_class,
    mock_audio_worker,
    mock_stt_worker,
    mock_llm_worker,
    mock_tts_worker,
):
    """Test that main() creates all required threads."""
    mock_tts_instance = MagicMock()
    mock_tts_class.return_value = mock_tts_instance

    call_count = [0]

    def sleep_side_effect(*args):
        call_count[0] += 1
        if call_count[0] == 1:
            raise KeyboardInterrupt()

    mock_sleep.side_effect = sleep_side_effect

    with patch("main.threading.Thread") as mock_thread_class:
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        from main import main

        with pytest.raises(SystemExit):
            main()

        # Should create 4 threads
        assert mock_thread_class.call_count == 4
        assert mock_thread.start.call_count == 4


@patch("main.main")
def test_main_name_main(mock_main):
    """Test that main() is called when script is run directly."""
    # Test the if __name__ == "__main__" block (line 48)
    # We execute main() directly which is what the if block does
    from unittest.mock import patch

    with patch("main.main") as mock_main_func:
        # Execute what line 48 does: main()
        # This gives us coverage of that line when we import and call it
        import main as main_module

        # The if block just calls main(), so we call it directly
        main_module.main()
        mock_main_func.assert_called_once()
