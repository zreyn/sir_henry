"""Unit tests for audio_worker.py module."""
import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import queue


@patch('audio_worker.playback_audio_queue')
@patch('audio_worker.interrupt_event')
@patch('audio_worker.is_speaking')
@patch('audio_worker.sd.play')
@patch('audio_worker.sd.wait')
@patch('audio_worker.logger')
def test_audio_worker_success(
    mock_logger,
    mock_sd_wait,
    mock_sd_play,
    mock_is_speaking,
    mock_interrupt,
    mock_playback_queue,
):
    """Test audio_worker() with successful playback."""
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_rate = 24000
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return (audio_data, sample_rate)
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return (audio_data, sample_rate)
    
    mock_playback_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False
    
    from audio_worker import audio_worker
    
    with pytest.raises(KeyboardInterrupt):
        audio_worker()
    
    mock_is_speaking.set.assert_called_once()
    mock_sd_play.assert_called_once_with(audio_data, sample_rate)
    mock_sd_wait.assert_called_once()
    mock_is_speaking.clear.assert_called_once()


@patch('audio_worker.playback_audio_queue')
@patch('audio_worker.interrupt_event')
@patch('audio_worker.is_speaking')
@patch('audio_worker.sd.play')
@patch('audio_worker.sd.wait')
@patch('audio_worker.logger')
def test_audio_worker_interrupt(
    mock_logger,
    mock_sd_wait,
    mock_sd_play,
    mock_is_speaking,
    mock_interrupt,
    mock_playback_queue,
):
    """Test audio_worker() when interrupted."""
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_rate = 24000
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return (audio_data, sample_rate)
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return (audio_data, sample_rate)
    
    mock_playback_queue.get.side_effect = queue_get
    # First call: interrupt is set, so it should drain and continue
    # Second call: raise KeyboardInterrupt to break the loop
    mock_interrupt.is_set.side_effect = lambda: call_count[0] == 1
    mock_playback_queue.empty.return_value = False
    mock_playback_queue.get_nowait.side_effect = queue.Empty()
    
    from audio_worker import audio_worker
    
    with pytest.raises(KeyboardInterrupt):
        audio_worker()
    
    # Should drain queue and not play
    mock_sd_play.assert_not_called()
    mock_is_speaking.set.assert_not_called()


@patch('audio_worker.playback_audio_queue')
@patch('audio_worker.interrupt_event')
@patch('audio_worker.is_speaking')
@patch('audio_worker.sd.play')
@patch('audio_worker.sd.wait')
@patch('audio_worker.logger')
def test_audio_worker_drain_queue(
    mock_logger,
    mock_sd_wait,
    mock_sd_play,
    mock_is_speaking,
    mock_interrupt,
    mock_playback_queue,
):
    """Test audio_worker() drains queue when interrupted."""
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_rate = 24000
    
    call_count = [0]
    get_nowait_count = [0]
    
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return (audio_data, sample_rate)
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return (audio_data, sample_rate)
    
    def queue_get_nowait():
        get_nowait_count[0] += 1
        if get_nowait_count[0] == 1:
            return (audio_data, sample_rate)
        raise queue.Empty()
    
    mock_playback_queue.get.side_effect = queue_get
    mock_interrupt.is_set.side_effect = lambda: call_count[0] == 1
    
    # Mock queue.empty() and get_nowait()
    empty_call_count = [0]
    def queue_empty():
        empty_call_count[0] += 1
        # Return False first time (queue not empty), then True
        return empty_call_count[0] > 1
    
    mock_playback_queue.empty.side_effect = queue_empty
    mock_playback_queue.get_nowait.side_effect = queue_get_nowait
    
    from audio_worker import audio_worker
    
    with pytest.raises(KeyboardInterrupt):
        audio_worker()
    
    # Should attempt to drain queue
    assert mock_playback_queue.get_nowait.call_count >= 1


@patch('audio_worker.playback_audio_queue')
@patch('audio_worker.interrupt_event')
@patch('audio_worker.is_speaking')
@patch('audio_worker.sd.play')
@patch('audio_worker.sd.wait')
@patch('audio_worker.logger')
def test_audio_worker_playback_error(
    mock_logger,
    mock_sd_wait,
    mock_sd_play,
    mock_is_speaking,
    mock_interrupt,
    mock_playback_queue,
):
    """Test audio_worker() handles playback errors."""
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_rate = 24000
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return (audio_data, sample_rate)
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return (audio_data, sample_rate)
    
    mock_playback_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False
    mock_sd_play.side_effect = Exception("Playback error")
    
    from audio_worker import audio_worker
    
    with pytest.raises(KeyboardInterrupt):
        audio_worker()
    
    mock_logger.error.assert_called_once_with("Audio Playback Error: Playback error")
    # Should still clear is_speaking in finally block
    mock_is_speaking.clear.assert_called_once()


@patch('audio_worker.playback_audio_queue')
@patch('audio_worker.interrupt_event')
@patch('audio_worker.is_speaking')
@patch('audio_worker.sd.play')
@patch('audio_worker.sd.wait')
@patch('audio_worker.logger')
def test_audio_worker_wait_error(
    mock_logger,
    mock_sd_wait,
    mock_sd_play,
    mock_is_speaking,
    mock_interrupt,
    mock_playback_queue,
):
    """Test audio_worker() handles wait errors."""
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    sample_rate = 24000
    
    call_count = [0]
    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return (audio_data, sample_rate)
        elif call_count[0] == 2:
            raise KeyboardInterrupt()
        return (audio_data, sample_rate)
    
    mock_playback_queue.get.side_effect = queue_get
    mock_interrupt.is_set.return_value = False
    mock_sd_wait.side_effect = Exception("Wait error")
    
    from audio_worker import audio_worker
    
    with pytest.raises(KeyboardInterrupt):
        audio_worker()
    
    mock_logger.error.assert_called_once_with("Audio Playback Error: Wait error")
    mock_is_speaking.clear.assert_called_once()
