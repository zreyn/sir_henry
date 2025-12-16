"""
Sir Henry - LiveKit Voice Agent

A voice AI agent using LiveKit Agents framework with custom F5-TTS and Faster-Whisper plugins.
"""

import logging
import datetime

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext, JobProcess, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from config import (
    logger,
    SYSTEM_PROMPT,
    GREETING,
    REF_AUDIO_PATH,
    REF_TEXT,
    SPEED,
    DEVICE,
    STT_DEVICE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)


# Load environment variables
load_dotenv()


class VoiceAgent(Agent):
    """The voice AI agent."""

    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)

    async def on_enter(self):
        """Called when the agent enters a session."""
        # Generate an initial greeting
        self.session.generate_reply(
            instructions=f"Introduce yourself briefly with this greeting: '{GREETING}'."
        )

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"


def prewarm(proc: JobProcess):
    """
    Prewarm function to load models before the agent starts.
    This reduces latency when the first user connects.
    """
    # Import heavy plugins here to avoid multiprocessing spawn issues
    from plugins import F5TTS, FasterWhisperSTT

    logger.info("Prewarming models...")

    # Load Silero VAD
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Silero VAD loaded.")

    # Initialize F5-TTS (will lazy-load model on first use)
    proc.userdata["tts"] = F5TTS(
        ref_audio_path=REF_AUDIO_PATH,
        ref_text=REF_TEXT,
        speed=SPEED,
        device=DEVICE,
    )
    logger.info("F5-TTS initialized.")

    # Initialize Faster-Whisper STT
    proc.userdata["stt"] = FasterWhisperSTT(
        model_size="small",
        device=STT_DEVICE,
    )
    logger.info("Faster-Whisper STT initialized.")


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit agent.
    Called when a participant joins a room.
    """
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    # Connect to the LiveKit room
    await ctx.connect()

    # Create the agent session with our custom plugins
    session = AgentSession(
        stt=ctx.proc.userdata["stt"],
        tts=ctx.proc.userdata["tts"],
        vad=ctx.proc.userdata["vad"],
        llm=lk_openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=f"http://{OLLAMA_HOST}/v1",
        ),
    )

    # Start the session with our agent
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
    )

    logger.info("Agent session started.")


def main():
    """Run the LiveKit agent."""
    logger.info("Starting LiveKit Voice Agent...")

    # Create the agent server
    worker = agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name="voice-agent",
    )

    # Run the agent
    agents.cli.run_app(worker)


if __name__ == "__main__":
    main()
