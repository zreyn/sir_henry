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
                "numpy": MagicMock(),
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
                "numpy": MagicMock(),
            },
        ):
            with patch.dict("os.environ", {}, clear=True):
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
                "numpy": MagicMock(),
            },
        ):
            from main import main

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
        """Test that main publishes the microphone track with correct options."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value
        mock_audio_track = mock_livekit_rtc.LocalAudioTrack.create_audio_track.return_value

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": MagicMock(),
            },
        ):
            from main import main

            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            # Verify AudioSource was created
            mock_livekit_rtc.AudioSource.assert_called_once_with(48000, 1)

            # Verify LocalAudioTrack was created
            mock_livekit_rtc.LocalAudioTrack.create_audio_track.assert_called_once()
            call_args = mock_livekit_rtc.LocalAudioTrack.create_audio_track.call_args
            assert call_args[0][0] == "mic_track"

            # Verify TrackPublishOptions was created with SOURCE_MICROPHONE
            mock_livekit_rtc.TrackPublishOptions.assert_called_once()

            # Verify track was published
            mock_room_instance.local_participant.publish_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_registers_track_subscribed_handler(
        self, mock_env_vars, mock_livekit_rtc, mock_sounddevice
    ):
        """Test that main registers a track_subscribed event handler."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value
        registered_events = []

        def track_on_calls(event):
            registered_events.append(event)
            return lambda f: f

        mock_room_instance.on = track_on_calls

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": MagicMock(),
            },
        ):
            from main import main

            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            assert "track_subscribed" in registered_events

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
                "numpy": MagicMock(),
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
                "numpy": MagicMock(),
            },
        ):
            from main import main

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

            assert captured_handler is not None

            # Call the handler
            captured_handler(mock_track, mock_publication, mock_participant)

            # Let background task run
            await asyncio.sleep(0)

            # Verify AudioStream was created with the track
            mock_livekit_rtc.AudioStream.assert_called_once_with(mock_track)

    @pytest.mark.asyncio
    async def test_ignores_non_audio_tracks(self, mock_env_vars, mock_livekit_rtc, mock_sounddevice):
        """Test that non-audio tracks are ignored."""
        if "main" in sys.modules:
            del sys.modules["main"]

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
                "numpy": MagicMock(),
            },
        ):
            from main import main

            with patch("asyncio.sleep", side_effect=KeyboardInterrupt):
                try:
                    await main()
                except KeyboardInterrupt:
                    pass

            # Create mock video track
            mock_track = MagicMock()
            mock_track.kind = "video"  # Not audio
            mock_publication = MagicMock()
            mock_participant = MagicMock()
            mock_participant.identity = "test-participant"

            assert captured_handler is not None

            # Call the handler
            captured_handler(mock_track, mock_publication, mock_participant)

            # AudioStream should not be created for video tracks
            mock_livekit_rtc.AudioStream.assert_not_called()


class TestPlayAudioStream:
    """Test the _play_audio_stream function."""

    @pytest.mark.asyncio
    async def test_plays_audio_frames(self, mock_sounddevice):
        """Test that audio frames are played through sounddevice."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_rtc = MagicMock()

        # Create mock audio stream that yields one frame
        class MockAudioStream:
            def __aiter__(self):
                async def gen():
                    mock_frame_event = MagicMock()
                    mock_frame_event.frame = MagicMock()
                    mock_frame_event.frame.data = b"\x00\x01" * 960
                    yield mock_frame_event
                return gen()

        mock_rtc.AudioStream = MockAudioStream

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_rtc),
                "livekit.rtc": mock_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": MagicMock(),
            },
        ):
            from main import _play_audio_stream

            audio_stream = MockAudioStream()
            await _play_audio_stream(audio_stream)

            # Verify sounddevice stream was used
            mock_sounddevice.RawOutputStream.assert_called_once_with(
                samplerate=48000,
                channels=1,
                dtype="int16",
                blocksize=960,
            )
            mock_stream = mock_sounddevice.RawOutputStream.return_value
            mock_stream.start.assert_called_once()
            mock_stream.write.assert_called()
            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_stream_error(self, mock_sounddevice):
        """Test that errors during streaming are handled gracefully."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_rtc = MagicMock()

        # Create mock audio stream that raises an exception
        class MockAudioStreamWithError:
            async def __aiter__(self):
                raise RuntimeError("Stream error")
                yield  # pragma: no cover

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_rtc),
                "livekit.rtc": mock_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": MagicMock(),
            },
        ):
            from main import _play_audio_stream

            audio_stream = MockAudioStreamWithError()
            # Should not raise, error is caught internally
            await _play_audio_stream(audio_stream)

            # Stream should still be cleaned up
            mock_stream = mock_sounddevice.RawOutputStream.return_value
            mock_stream.stop.assert_called_once()
            mock_stream.close.assert_called_once()


class TestCaptureMicrophone:
    """Test the _capture_microphone function."""

    @pytest.mark.asyncio
    async def test_creates_input_stream(self, mock_sounddevice, mock_livekit_rtc):
        """Test that microphone capture creates an input stream."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_audio_source = MagicMock()
        mock_audio_source.capture_frame = AsyncMock()

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": MagicMock(),
            },
        ):
            from main import _capture_microphone

            # Run capture briefly then cancel
            task = asyncio.create_task(_capture_microphone(mock_audio_source, gain=3.0))
            await asyncio.sleep(0.01)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify InputStream was created with correct parameters
            mock_sounddevice.InputStream.assert_called_once()
            call_kwargs = mock_sounddevice.InputStream.call_args[1]
            assert call_kwargs["samplerate"] == 48000
            assert call_kwargs["channels"] == 1
            assert call_kwargs["dtype"] == "float32"
            assert call_kwargs["blocksize"] == 960
            assert "callback" in call_kwargs

    @pytest.mark.asyncio
    async def test_audio_callback_processes_audio(self, mock_sounddevice, mock_livekit_rtc):
        """Test that the audio callback processes and sends audio frames."""
        if "main" in sys.modules:
            del sys.modules["main"]

        import numpy as np

        mock_audio_source = MagicMock()
        mock_audio_source.capture_frame = AsyncMock()

        captured_callback = None

        def capture_input_stream(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("callback")
            return MagicMock()

        mock_sounddevice.InputStream = capture_input_stream

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import _capture_microphone

            # Run capture briefly to set up callback
            task = asyncio.create_task(_capture_microphone(mock_audio_source, gain=3.0))
            await asyncio.sleep(0.01)

            # Verify callback was captured
            assert captured_callback is not None

            # Simulate audio input
            mock_indata = np.zeros((960, 1), dtype=np.float32)
            mock_indata[0, 0] = 0.5  # Some audio data

            # Call the callback
            captured_callback(mock_indata, 960, None, None)

            # Give time for async operations
            await asyncio.sleep(0.01)

            # Verify AudioFrame was created
            mock_livekit_rtc.AudioFrame.assert_called()

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_audio_callback_handles_status(self, mock_sounddevice, mock_livekit_rtc):
        """Test that the audio callback logs status messages."""
        if "main" in sys.modules:
            del sys.modules["main"]

        import numpy as np

        mock_audio_source = MagicMock()
        mock_audio_source.capture_frame = AsyncMock()

        captured_callback = None

        def capture_input_stream(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("callback")
            return MagicMock()

        mock_sounddevice.InputStream = capture_input_stream

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
            },
        ):
            from main import _capture_microphone

            task = asyncio.create_task(_capture_microphone(mock_audio_source, gain=3.0))
            await asyncio.sleep(0.01)

            assert captured_callback is not None

            # Simulate audio input with status
            mock_indata = np.zeros((960, 1), dtype=np.float32)
            
            # Call callback with a status (should log warning but not fail)
            captured_callback(mock_indata, 960, None, "input overflow")

            await asyncio.sleep(0.01)

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_audio_callback_handles_error(self, mock_sounddevice, mock_livekit_rtc):
        """Test that the audio callback handles errors gracefully."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_audio_source = MagicMock()
        mock_audio_source.capture_frame = AsyncMock()

        captured_callback = None

        def capture_input_stream(**kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get("callback")
            return MagicMock()

        mock_sounddevice.InputStream = capture_input_stream

        # Make AudioFrame raise an error
        mock_livekit_rtc.AudioFrame = MagicMock(side_effect=RuntimeError("Frame error"))

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc),
                "livekit.rtc": mock_livekit_rtc,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": MagicMock(),
            },
        ):
            from main import _capture_microphone
            import numpy as np

            task = asyncio.create_task(_capture_microphone(mock_audio_source, gain=3.0))
            await asyncio.sleep(0.01)

            assert captured_callback is not None

            # Simulate audio input - callback should handle the error gracefully
            mock_indata = np.zeros((960, 1), dtype=np.float32)
            
            # This should not raise, error is caught internally
            captured_callback(mock_indata, 960, None, None)

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
