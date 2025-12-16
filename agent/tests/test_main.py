"""Unit tests for main.py module."""

import datetime
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# Mock all heavy dependencies before importing main
@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all heavy dependencies."""
    # Create mock modules
    mock_livekit = MagicMock()
    mock_livekit.agents = MagicMock()
    mock_livekit.agents.AgentSession = MagicMock
    mock_livekit.agents.Agent = MagicMock
    mock_livekit.agents.JobContext = MagicMock
    mock_livekit.agents.JobProcess = MagicMock
    mock_livekit.agents.RunContext = MagicMock
    mock_livekit.agents.llm = MagicMock()
    mock_livekit.agents.llm.function_tool = lambda f: f
    mock_livekit.agents.WorkerOptions = MagicMock
    mock_livekit.agents.cli = MagicMock()

    mock_lk_openai = MagicMock()
    mock_lk_openai.LLM = MagicMock()
    mock_lk_openai.LLM.with_ollama = MagicMock(return_value=MagicMock())

    mock_silero = MagicMock()
    mock_silero.VAD = MagicMock()
    mock_silero.VAD.load = MagicMock(return_value=MagicMock())

    mock_plugins = MagicMock()
    mock_plugins.F5TTS = MagicMock(return_value=MagicMock())
    mock_plugins.FasterWhisperSTT = MagicMock(return_value=MagicMock())

    with patch.dict(
        "sys.modules",
        {
            "livekit": mock_livekit,
            "livekit.agents": mock_livekit.agents,
            "livekit.agents.llm": mock_livekit.agents.llm,
            "livekit.plugins": MagicMock(),
            "livekit.plugins.openai": mock_lk_openai,
            "livekit.plugins.silero": mock_silero,
            "dotenv": MagicMock(),
            "plugins": mock_plugins,
        },
    ):
        # Also mock config to avoid torch import issues
        mock_config = MagicMock()
        mock_config.logger = MagicMock()
        mock_config.SYSTEM_PROMPT = "Test prompt"
        mock_config.GREETING = "Hello"
        mock_config.REF_AUDIO_PATH = "/path/to/ref.wav"
        mock_config.REF_TEXT = "Reference text"
        mock_config.SPEED = 1.0
        mock_config.DEVICE = "cpu"
        mock_config.STT_DEVICE = "cpu"
        mock_config.OLLAMA_HOST = "localhost:11434"
        mock_config.OLLAMA_MODEL = "llama3.2:3b"

        with patch.dict("sys.modules", {"config": mock_config}):
            yield {
                "livekit": mock_livekit,
                "lk_openai": mock_lk_openai,
                "silero": mock_silero,
                "plugins": mock_plugins,
                "config": mock_config,
            }


class TestVoiceAgent:
    """Test VoiceAgent class."""

    def test_voice_agent_init(self, mock_dependencies):
        """Test VoiceAgent initialization."""
        # Import after mocking
        if "main" in sys.modules:
            del sys.modules["main"]

        # Need to create a proper Agent base class mock
        mock_agent_base = MagicMock()
        mock_dependencies["livekit"].agents.Agent = mock_agent_base

        from main import VoiceAgent

        agent = VoiceAgent()
        # Agent should be created with instructions
        assert agent is not None

    @pytest.mark.asyncio
    async def test_voice_agent_on_enter(self, mock_dependencies):
        """Test VoiceAgent.on_enter method."""
        if "main" in sys.modules:
            del sys.modules["main"]

        from main import VoiceAgent

        agent = VoiceAgent()
        agent.session = MagicMock()
        agent.session.generate_reply = MagicMock()

        await agent.on_enter()

        agent.session.generate_reply.assert_called_once()
        call_args = agent.session.generate_reply.call_args
        assert "instructions" in call_args.kwargs
        assert "Hello" in call_args.kwargs["instructions"]

    @pytest.mark.asyncio
    async def test_get_current_date_and_time(self, mock_dependencies):
        """Test get_current_date_and_time tool."""
        if "main" in sys.modules:
            del sys.modules["main"]

        from main import VoiceAgent

        agent = VoiceAgent()
        mock_context = MagicMock()

        result = await agent.get_current_date_and_time(mock_context)

        assert "current date and time" in result.lower()
        # Should contain date format elements
        assert any(
            month in result
            for month in [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        )


class TestPrewarm:
    """Test prewarm function."""

    def test_prewarm_loads_models(self, mock_dependencies):
        """Test that prewarm loads all required models."""
        if "main" in sys.modules:
            del sys.modules["main"]

        from main import prewarm

        mock_proc = MagicMock()
        mock_proc.userdata = {}

        prewarm(mock_proc)

        # Should have loaded VAD, TTS, and STT
        assert "vad" in mock_proc.userdata
        assert "tts" in mock_proc.userdata
        assert "stt" in mock_proc.userdata


class TestEntrypoint:
    """Test entrypoint function."""

    @pytest.mark.asyncio
    async def test_entrypoint_connects_and_starts_session(self, mock_dependencies):
        """Test that entrypoint connects to room and starts session."""
        if "main" in sys.modules:
            del sys.modules["main"]

        from main import entrypoint

        mock_ctx = MagicMock()
        mock_ctx.room = MagicMock()
        mock_ctx.room.name = "test-room"
        mock_ctx.connect = AsyncMock()
        mock_ctx.proc = MagicMock()
        mock_ctx.proc.userdata = {
            "vad": MagicMock(),
            "tts": MagicMock(),
            "stt": MagicMock(),
        }

        # Mock AgentSession
        mock_session = MagicMock()
        mock_session.start = AsyncMock()

        with patch("main.AgentSession", return_value=mock_session):
            await entrypoint(mock_ctx)

        mock_ctx.connect.assert_called_once()
        mock_session.start.assert_called_once()


class TestMain:
    """Test main function."""

    def test_main_creates_worker_and_runs(self, mock_dependencies):
        """Test that main creates worker options and runs."""
        if "main" in sys.modules:
            del sys.modules["main"]

        from main import main

        mock_worker_options = MagicMock()
        mock_dependencies["livekit"].agents.WorkerOptions = MagicMock(
            return_value=mock_worker_options
        )

        main()

        # Should create WorkerOptions with correct arguments
        mock_dependencies["livekit"].agents.WorkerOptions.assert_called_once()
        call_kwargs = mock_dependencies["livekit"].agents.WorkerOptions.call_args.kwargs
        assert "entrypoint_fnc" in call_kwargs
        assert "prewarm_fnc" in call_kwargs
        assert call_kwargs["agent_name"] == "voice-agent"

        # Should run the app
        mock_dependencies["livekit"].agents.cli.run_app.assert_called_once_with(
            mock_worker_options
        )
