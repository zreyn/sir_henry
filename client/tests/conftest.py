"""Pytest configuration and fixtures."""

import os
import sys
from unittest.mock import MagicMock, AsyncMock

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_TOKEN", "test-token")


@pytest.fixture
def mock_livekit_rtc():
    """Mock livekit.rtc module."""
    mock_rtc = MagicMock()

    # Mock Room
    mock_room = MagicMock()
    mock_room.connect = AsyncMock()
    mock_room.disconnect = AsyncMock()
    mock_room.on = MagicMock(side_effect=lambda event: lambda f: f)
    mock_room.local_participant = MagicMock()
    mock_room.local_participant.publish_track = AsyncMock()
    mock_rtc.Room = MagicMock(return_value=mock_room)

    # Mock TrackKind
    mock_rtc.TrackKind = MagicMock()
    mock_rtc.TrackKind.KIND_AUDIO = "audio"

    # Mock AudioStream
    class _MockAudioStream:
        def __init__(self, track):
            self.track = track

        def __aiter__(self):
            async def gen():
                mock_frame = MagicMock()
                mock_frame.data = b"\x00\x00" * 960
                yield mock_frame

            return gen()

    mock_rtc.AudioStream = MagicMock(side_effect=_MockAudioStream)

    # Mock LocalAudioTrack
    mock_audio_track = MagicMock()
    mock_rtc.LocalAudioTrack = MagicMock()
    mock_rtc.LocalAudioTrack.create_microphone_track = MagicMock(
        return_value=mock_audio_track
    )

    return mock_rtc


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module."""
    mock_sd = MagicMock()
    mock_stream = MagicMock()
    mock_sd.RawOutputStream = MagicMock(return_value=mock_stream)
    return mock_sd
