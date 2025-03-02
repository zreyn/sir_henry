from src import sirhenry

import unittest
from unittest.mock import patch, MagicMock, Mock, call
import io
import pytest


@patch("src.sirhenry.Agent")
@patch("src.sirhenry.sr.Recognizer")
@patch("src.sirhenry.ElevenLabs")
def test_setup_sir_henry(mock_elevenlabs, mock_recognizer, mock_agent):
    """Test the setup_sir_henry function."""

    agent, tts_engine, listener = sirhenry.setup_sir_henry()

    # Assertions
    assert agent == mock_agent.return_value
    assert tts_engine == mock_elevenlabs.return_value
    assert listener == mock_recognizer.return_value

    mock_agent.assert_called_once()
    mock_recognizer.assert_called_once()
    mock_elevenlabs.assert_called_once()


# def test_text_loop(mock_run_sync, mock_speech_to_text, mock_text_to_speech, capsys):
#     """Test the text_loop function."""
#     # Setup mock objects
#     mock_agent = MagicMock()
#     mock_tts_engine = MagicMock()
#     mock_listener = MagicMock()

#     mock_run_sync.side_effect = [
#         MagicMock(data="Ahoy there!"),  # Initial response
#         MagicMock(data="I be Sir Henry!"),  # Response to user input
#     ]
#     mock_speech_to_text.side_effect = [
#         "Hello",
#         sr.UnknownValueError,
#         Exception("Test Exception"),
#         "Bye",
#         StopIteration,  # Add StopIteration
#     ]

#     with pytest.raises(StopIteration):
#         sirhenry.text_loop(mock_agent, mock_tts_engine, mock_listener)

#     # Check that agent.run_sync was called correctly.
#     mock_run_sync.assert_has_calls([call(""), call("Hello")])

#     # Check that speech_to_text and text_to_speech were called.
#     mock_speech_to_text.assert_called()
#     mock_text_to_speech.assert_has_calls(
#         [
#             call("Ahoy there!", mock_tts_engine),
#             call("Sorry, I didn't catch that.", mock_tts_engine),
#             call("I be Sir Henry!", mock_tts_engine),
#         ]
#     )

#     # Mock speech_to_text to raise StopIteration on 3rd call
#     mock_speech_to_text.side_effect = ["Hi", "How are you?", StopIteration]
#     with pytest.raises(StopIteration):
#         sirhenry.text_loop(mock_agent, mock_tts_engine, mock_listener)


# def test_text_to_speech(mock_convert_as_stream, mock_stream, capsys):
#     """Test the text_to_speech function."""

#     mock_tts_engine = MagicMock()
#     mock_audio_stream = io.BytesIO(b"fake audio data")  # Use BytesIO
#     mock_convert_as_stream.return_value = mock_audio_stream

#     sirhenry.text_to_speech("Test text", mock_tts_engine)

#     # Assert calls
#     mock_convert_as_stream.assert_called_once_with(
#         voice_id="PPzYpIqttlTYA83688JI",
#         model_id="eleven_multilingual_v2",
#         text="Test text",
#     )
#     mock_stream.assert_called_once_with(mock_audio_stream)

#     # Test exception handling
#     mock_convert_as_stream.side_effect = Exception("TTS Error")
#     sirhenry.text_to_speech("Test text", mock_tts_engine)
#     captured = capsys.readouterr()
#     assert "Error during TTS conversion: TTS Error" in captured.out


# def test_speech_to_text(
#     mock_listen,
#     mock_adjust_for_ambient_noise,
#     mock_recognize_whisper,
#     mock_microphone,
#     capsys,
# ):
#     """Test the speech_to_text function."""
#     mock_listener = MagicMock()
#     mock_source = MagicMock()
#     mock_voice = MagicMock()
#     mock_microphone.return_value.__enter__.return_value = (
#         mock_source  # Mock context manager
#     )

#     mock_listen.return_value = mock_voice
#     mock_recognize_whisper.return_value = "test input"

#     result = sirhenry.speech_to_text(mock_listener)

#     # Assertions
#     mock_microphone.assert_called_once()  # Check microphone context manager
#     mock_adjust_for_ambient_noise.assert_called_once_with(mock_source)
#     mock_listen.assert_called_once_with(mock_source)
#     mock_recognize_whisper.assert_called_once_with(mock_voice)
#     assert result == "test input"
#     captured = capsys.readouterr()  # Capture the print
#     assert "Heard: test input" in captured.out

#     # Test with uppercase conversion.
#     mock_recognize_whisper.return_value = "TEST INPUT"
#     result = sirhenry.speech_to_text(mock_listener)
#     assert result == "test input"
#     captured = capsys.readouterr()
#     assert "Heard: test input" in captured.out


# def test_speech_to_text_microphone_exception(mock_microphone):
#     """Test speech_to_text handles Microphone errors gracefully."""

#     mock_listener = MagicMock()
#     mock_microphone.side_effect = sr.RequestError("Microphone error")

#     with pytest.raises(sr.RequestError) as excinfo:
#         sirhenry.speech_to_text(mock_listener)
#     assert str(excinfo.value) == "Microphone error"
