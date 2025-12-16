"""Unit tests for client main.py module."""

import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class TestClientConfiguration:
    """Test client configuration loading."""

    def test_loads_environment_variables(self, mock_env_vars, mock_livekit_rtc, mock_sounddevice):
        """Test that URL and TOKEN are loaded from environment."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import URL, TOKEN

            assert URL == "ws://localhost:7880"
            assert TOKEN == "test-token"

    def test_handles_missing_environment_variables(self, mock_livekit_rtc, mock_sounddevice):
        """Test behavior when environment variables are not set."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            # Clear environment
            with patch.dict("os.environ", {}, clear=True):
                # Re-import to get fresh values
                if "main" in sys.modules:
                    del sys.modules["main"]

                from main import URL, TOKEN

                assert URL is None
                assert TOKEN is None


class TestMain:
    """Test main function."""

    @pytest.mark.asyncio
    async def test_connects_to_room(self, mock_env_vars, mock_livekit_rtc, mock_sounddevice):
        """Test that main connects to the LiveKit room."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import main

            # Patch asyncio.sleep to break the infinite loop after first iteration
            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            mock_room_instance.connect.assert_called_once_with(
                "ws://localhost:7880", "test-token"
            )

    @pytest.mark.asyncio
    async def test_publishes_microphone_track(self, mock_env_vars, mock_livekit_rtc, mock_sounddevice):
        """Test that main publishes the microphone track."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value
        mock_audio_track = (
            mock_livekit_rtc.LocalAudioTrack.create_microphone_track.return_value
        )

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import main

            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            mock_livekit_rtc.LocalAudioTrack.create_microphone_track.assert_called_once_with(
                "mic_track"
            )
            mock_room_instance.local_participant.publish_track.assert_called_once_with(
                mock_audio_track
            )

    @pytest.mark.asyncio
    async def test_registers_track_subscribed_handler(
        self, mock_env_vars, mock_livekit_rtc, mock_sounddevice
    ):
        """Test that main registers a track_subscribed event handler."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import main

            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            # Check that room.on was called with "track_subscribed"
            mock_room_instance.on.assert_called_with("track_subscribed")

    @pytest.mark.asyncio
    async def test_disconnects_on_keyboard_interrupt(
        self, mock_env_vars, mock_livekit_rtc, mock_sounddevice
    ):
        """Test that main disconnects from room on KeyboardInterrupt."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import main

            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            mock_room_instance.disconnect.assert_called_once()


class TestTrackSubscribedHandler:
    """Test the track_subscribed event handler."""

    @pytest.mark.asyncio
    async def test_handles_audio_track(self, mock_env_vars, mock_livekit_rtc, mock_sounddevice):
        """Test that audio tracks are handled correctly."""
        if "main" in sys.modules:
            del sys.modules["main"]

        # Capture the handler when room.on is called
        captured_handler = None

        def capture_on(event):
            def decorator(f):
                nonlocal captured_handler
                if event == "track_subscribed":
                    captured_handler = f
                return f

            return decorator

        mock_room_instance = mock_livekit_rtc.Room.return_value
        mock_room_instance.on = capture_on

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import main

            # Run main to register the handler
            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            # Create mock track and participant
            mock_track = MagicMock()
            mock_track.kind = mock_livekit_rtc.TrackKind.KIND_AUDIO
            mock_publication = MagicMock()
            mock_participant = MagicMock()
            mock_participant.identity = "test-participant"

            # The handler should have been captured when main() ran
            assert captured_handler is not None

            # Call the handler - should not raise
            captured_handler(mock_track, mock_publication, mock_participant)

            # Let background task run
            await asyncio.sleep(0)

            # Verify AudioStream was created with the track
            mock_livekit_rtc.AudioStream.assert_called_once_with(mock_track)

            # Verify sounddevice stream usage
            mock_sounddevice.RawOutputStream.assert_called_once()
            mock_stream = mock_sounddevice.RawOutputStream.return_value
            mock_stream.start.assert_called_once()
            mock_stream.write.assert_called()
            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()
