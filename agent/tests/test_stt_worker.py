"""Unit tests for stt_worker.py module."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import torch


@patch("stt_worker.torch.hub.load")
@patch("faster_whisper.WhisperModel")
@patch("stt_worker.logger")
def test_load_models(mock_logger, mock_whisper_model, mock_torch_hub):
    """Test load_models() function."""
    mock_vad_model = MagicMock()
    mock_utils = MagicMock()
    mock_torch_hub.return_value = (mock_vad_model, mock_utils)

    mock_whisper_instance = MagicMock()
    mock_whisper_model.return_value = mock_whisper_instance

    from stt_worker import load_models

    vad_model, whisper_model = load_models()

    assert vad_model == mock_vad_model
    assert whisper_model == mock_whisper_instance
    mock_torch_hub.assert_called_once_with(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    mock_whisper_model.assert_called_once_with(
        "small", device="cpu", compute_type="int8"
    )
    mock_logger.info.assert_any_call("Loading Silero VAD...")
    mock_logger.info.assert_any_call("Loading Faster Whisper...")
    mock_logger.info.assert_any_call("Initializing Whisper on CPU...")
    mock_logger.info.assert_any_call("Models loaded. STT running on CPU.")


@patch("stt_worker.logger")
def test_audio_callback_no_status(mock_logger):
    """Test audio_callback() with no status."""
    from stt_worker import audio_callback, mic_audio_queue

    # Clear queue
    while not mic_audio_queue.empty():
        try:
            mic_audio_queue.get_nowait()
        except:
            break

    indata = np.array([[1, 2, 3]], dtype=np.int16)
    frames = 3
    time_info = {}
    status = None

    audio_callback(indata, frames, time_info, status)

    # Check that data was put in queue
    assert not mic_audio_queue.empty()
    queued_data = mic_audio_queue.get()
    np.testing.assert_array_equal(queued_data, indata.copy())
    mock_logger.error.assert_not_called()


@patch("stt_worker.logger")
def test_audio_callback_with_status(mock_logger):
    """Test audio_callback() with status error."""
    from stt_worker import audio_callback, mic_audio_queue

    # Clear queue
    while not mic_audio_queue.empty():
        try:
            mic_audio_queue.get_nowait()
        except:
            break

    indata = np.array([[1, 2, 3]], dtype=np.int16)
    frames = 3
    time_info = {}
    status = "Input overflow"

    audio_callback(indata, frames, time_info, status)

    mock_logger.error.assert_called_once_with("Audio callback status: Input overflow")
    # Data should still be queued
    assert not mic_audio_queue.empty()


@patch("stt_worker.torch.no_grad")
@patch("stt_worker.torch.from_numpy")
def test_is_speech_detected(mock_from_numpy, mock_no_grad):
    """Test is_speech() when speech is detected."""
    from stt_worker import is_speech, VAD_THRESHOLD

    mock_model = MagicMock()
    mock_model.return_value.item.return_value = 0.7  # Above threshold

    audio_chunk = np.array([1000, 2000, 3000], dtype=np.int16)
    mock_tensor = MagicMock()
    mock_from_numpy.return_value = mock_tensor
    mock_tensor.shape = (3,)
    mock_tensor.astype.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Mock the model call
    mock_model.return_value.item.return_value = 0.7

    result = is_speech(mock_model, audio_chunk)

    assert result is True
    mock_model.assert_called_once()


@patch("stt_worker.torch.no_grad")
@patch("stt_worker.torch.from_numpy")
def test_is_speech_not_detected(mock_from_numpy, mock_no_grad):
    """Test is_speech() when speech is not detected."""
    from stt_worker import is_speech, VAD_THRESHOLD

    mock_model = MagicMock()
    mock_model.return_value.item.return_value = 0.3  # Below threshold

    audio_chunk = np.array([100, 200, 300], dtype=np.int16)
    mock_tensor = MagicMock()
    mock_from_numpy.return_value = mock_tensor
    mock_tensor.shape = (3,)

    result = is_speech(mock_model, audio_chunk)

    assert result is False


@patch("stt_worker.torch.no_grad")
@patch("stt_worker.torch.from_numpy")
def test_is_speech_multidimensional(mock_from_numpy, mock_no_grad):
    """Test is_speech() with multidimensional audio."""
    from stt_worker import is_speech

    mock_model = MagicMock()
    mock_model.return_value.item.return_value = 0.6

    audio_chunk = np.array([[1000], [2000], [3000]], dtype=np.int16)
    mock_tensor = MagicMock()
    mock_tensor.shape = (3, 1)  # Multidimensional
    mock_tensor.squeeze.return_value = MagicMock()
    mock_tensor.squeeze.return_value.shape = (3,)
    mock_from_numpy.return_value = mock_tensor

    result = is_speech(mock_model, audio_chunk)

    mock_tensor.squeeze.assert_called_once()


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
def test_listen_basic_flow(
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() basic speech detection and transcription flow."""
    # Setup mocks
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    # Create a queue that will provide chunks
    chunk_queue = []
    chunk1 = np.array([[1000] * 512], dtype=np.int16)  # Speech chunk
    chunk2 = np.array([[100] * 512], dtype=np.int16)  # Silence chunk
    chunk3 = np.array([[50] * 512], dtype=np.int16)  # More silence

    # Simulate: speech detected -> triggered -> silence -> process
    # silence_limit_chunks = int(0.8 * (16000 / 512)) = 25
    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return chunk1  # Speech - triggers
        elif call_count[0] <= 26:  # 25 silence chunks to reach limit
            return chunk2  # Silence
        elif call_count[0] == 27:
            raise KeyboardInterrupt()
        return chunk2

    mock_mic_queue.get.side_effect = queue_get

    # Speech detection: first chunk is speech (triggers), rest are silence
    speech_calls = [0]

    def is_speech_side_effect(model, chunk):
        speech_calls[0] += 1
        if speech_calls[0] == 1:
            return True  # First chunk triggers
        return False  # All subsequent chunks are silence

    mock_is_speech_func.side_effect = is_speech_side_effect

    mock_is_speaking.is_set.return_value = False
    mock_playback_queue.empty.return_value = True

    # Mock whisper transcription
    mock_segment = MagicMock()
    mock_segment.text = "hello world"
    mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

    from stt_worker import listen

    # listen() catches KeyboardInterrupt and breaks, so it returns normally
    listen()

    mock_logger.info.assert_any_call("Listening... (Press Ctrl+C to stop)")
    mock_logger.info.assert_any_call("Speech Detected")
    mock_whisper_model.transcribe.assert_called()
    mock_prompt_queue.put.assert_called_once_with("hello world")


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
def test_listen_skip_when_speaking(
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() skips processing when system is speaking."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    chunk = np.array([[1000] * 512], dtype=np.int16)
    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] > 10:
            raise KeyboardInterrupt()
        return chunk

    mock_mic_queue.get.side_effect = queue_get
    mock_is_speaking.is_set.return_value = True  # System is speaking

    from stt_worker import listen

    # listen() catches KeyboardInterrupt and breaks, so it returns normally
    listen()

    # is_speech should not be called when system is speaking
    mock_is_speech_func.assert_not_called()


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
@patch("stt_worker.sd.stop")
def test_listen_interrupt_playback(
    mock_sd_stop,
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() interrupts playback when user speaks."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    chunk1 = np.array([[1000] * 512], dtype=np.int16)
    chunk2 = np.array([[100] * 512], dtype=np.int16)

    # silence_limit_chunks = 25
    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return chunk1  # Speech - triggers
        elif call_count[0] <= 26:  # 25 silence chunks
            return chunk2
        elif call_count[0] == 27:
            raise KeyboardInterrupt()
        return chunk2

    mock_mic_queue.get.side_effect = queue_get

    speech_calls = [0]

    def is_speech_side_effect(model, chunk):
        speech_calls[0] += 1
        return speech_calls[0] == 1

    mock_is_speech_func.side_effect = is_speech_side_effect
    mock_is_speaking.is_set.return_value = False
    mock_playback_queue.empty.return_value = False  # Playback is active

    mock_segment = MagicMock()
    mock_segment.text = "interrupt"
    mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

    from stt_worker import listen

    # listen() catches KeyboardInterrupt and breaks, so it returns normally
    listen()

    mock_interrupt_event.set.assert_called()
    mock_sd_stop.assert_called()


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
def test_listen_empty_transcription(
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() handles empty transcription."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    chunk1 = np.array([[1000] * 512], dtype=np.int16)
    chunk2 = np.array([[100] * 512], dtype=np.int16)

    # silence_limit_chunks = 25
    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return chunk1  # Speech - triggers
        elif call_count[0] <= 26:  # 25 silence chunks
            return chunk2
        elif call_count[0] == 27:
            raise KeyboardInterrupt()
        return chunk2

    mock_mic_queue.get.side_effect = queue_get

    speech_calls = [0]

    def is_speech_side_effect(model, chunk):
        speech_calls[0] += 1
        return speech_calls[0] == 1

    mock_is_speech_func.side_effect = is_speech_side_effect
    mock_is_speaking.is_set.return_value = False
    mock_playback_queue.empty.return_value = True

    # Empty transcription
    mock_whisper_model.transcribe.return_value = ([], MagicMock())

    from stt_worker import listen

    # listen() catches KeyboardInterrupt and breaks, so it returns normally
    listen()

    mock_logger.info.assert_any_call("(No text detected)")
    mock_prompt_queue.put.assert_not_called()


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
def test_listen_transcription_error(
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() handles transcription errors."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    chunk1 = np.array([[1000] * 512], dtype=np.int16)
    chunk2 = np.array([[100] * 512], dtype=np.int16)

    # silence_limit_chunks = 25
    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return chunk1  # Speech - triggers
        elif call_count[0] <= 26:  # 25 silence chunks
            return chunk2
        elif call_count[0] == 27:
            raise KeyboardInterrupt()
        return chunk2

    mock_mic_queue.get.side_effect = queue_get

    speech_calls = [0]

    def is_speech_side_effect(model, chunk):
        speech_calls[0] += 1
        return speech_calls[0] == 1

    mock_is_speech_func.side_effect = is_speech_side_effect
    mock_is_speaking.is_set.return_value = False
    mock_playback_queue.empty.return_value = True

    mock_whisper_model.transcribe.side_effect = Exception("Transcription error")

    from stt_worker import listen

    # listen() catches KeyboardInterrupt and breaks, so it returns normally
    listen()

    mock_logger.error.assert_any_call("Error during transcription: Transcription error")


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
def test_listen_general_exception(
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() handles general exceptions."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    mock_mic_queue.get.side_effect = Exception("General error")

    from stt_worker import listen

    listen()  # Should break out of loop on exception

    mock_logger.error.assert_any_call("Error in listen loop: General error")


@patch("stt_worker.listen")
def test_stt_worker(mock_listen):
    """Test stt_worker() function."""
    from stt_worker import stt_worker

    stt_worker()

    mock_listen.assert_called_once()


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
@patch("stt_worker.sd.stop")
def test_listen_speech_during_triggered(
    mock_sd_stop,
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() when speech is detected again while triggered (resets silence_counter)."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    chunk1 = np.array([[1000] * 512], dtype=np.int16)
    chunk2 = np.array([[100] * 512], dtype=np.int16)

    # silence_limit_chunks = 25
    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return chunk1  # Speech - triggers
        elif call_count[0] == 2:
            return chunk1  # Speech again - resets silence_counter
        elif call_count[0] <= 27:  # 25 silence chunks after second speech
            return chunk2
        elif call_count[0] == 28:
            raise KeyboardInterrupt()
        return chunk2

    mock_mic_queue.get.side_effect = queue_get

    speech_calls = [0]

    def is_speech_side_effect(model, chunk):
        speech_calls[0] += 1
        # First and second chunks are speech, rest are silence
        return speech_calls[0] <= 2

    mock_is_speech_func.side_effect = is_speech_side_effect
    mock_is_speaking.is_set.return_value = False
    mock_playback_queue.empty.return_value = True

    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

    from stt_worker import listen

    listen()

    # Should have processed once
    mock_whisper_model.transcribe.assert_called()


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
@patch("stt_worker.sd.stop")
def test_listen_sd_stop_exception(
    mock_sd_stop,
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() handles sd.stop() exception."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    chunk1 = np.array([[1000] * 512], dtype=np.int16)
    chunk2 = np.array([[100] * 512], dtype=np.int16)

    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] == 1:
            return chunk1
        elif call_count[0] <= 26:
            return chunk2
        elif call_count[0] == 27:
            raise KeyboardInterrupt()
        return chunk2

    mock_mic_queue.get.side_effect = queue_get

    speech_calls = [0]

    def is_speech_side_effect(model, chunk):
        speech_calls[0] += 1
        return speech_calls[0] == 1

    mock_is_speech_func.side_effect = is_speech_side_effect
    mock_is_speaking.is_set.return_value = False
    mock_playback_queue.empty.return_value = False  # Playback active
    mock_sd_stop.side_effect = Exception("Stop error")  # sd.stop() raises exception

    mock_segment = MagicMock()
    mock_segment.text = "test"
    mock_whisper_model.transcribe.return_value = ([mock_segment], MagicMock())

    from stt_worker import listen

    # Should not raise, exception is caught
    listen()

    mock_sd_stop.assert_called()


@patch("stt_worker.load_models")
@patch("stt_worker.sd.InputStream")
@patch("stt_worker.is_speech")
@patch("stt_worker.logger")
@patch("stt_worker.mic_audio_queue")
@patch("stt_worker.is_speaking")
@patch("stt_worker.interrupt_event")
@patch("stt_worker.sentence_queue")
@patch("stt_worker.playback_audio_queue")
@patch("stt_worker.prompt_queue")
def test_listen_pre_roll_buffer_management(
    mock_prompt_queue,
    mock_playback_queue,
    mock_sentence_queue,
    mock_interrupt_event,
    mock_is_speaking,
    mock_mic_queue,
    mock_logger,
    mock_is_speech_func,
    mock_sd_input,
    mock_load_models,
):
    """Test listen() manages pre_roll_buffer correctly."""
    mock_vad_model = MagicMock()
    mock_whisper_model = MagicMock()
    mock_load_models.return_value = (mock_vad_model, mock_whisper_model)

    mock_context = MagicMock()
    mock_sd_input.return_value.__enter__.return_value = mock_context
    mock_sd_input.return_value.__exit__.return_value = None

    chunk = np.array([[100] * 512], dtype=np.int16)

    # pre_roll_chunks = int((200/1000) * (16000/512)) = int(0.2 * 31.25) = int(6.25) = 6
    # Need more than 6 chunks to test pop(0)
    call_count = [0]

    def queue_get():
        call_count[0] += 1
        if call_count[0] <= 10:  # More than pre_roll_chunks
            return chunk
        elif call_count[0] == 11:
            raise KeyboardInterrupt()
        return chunk

    mock_mic_queue.get.side_effect = queue_get
    mock_is_speech_func.return_value = False  # All silence, not triggered

    mock_is_speaking.is_set.return_value = False
    mock_playback_queue.empty.return_value = True

    from stt_worker import listen

    listen()

    # Should have processed chunks without triggering
    assert (
        mock_is_speech_func.call_count >= 7
    )  # At least 7 calls to test buffer management
