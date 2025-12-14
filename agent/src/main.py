"""
Sir Henry - LiveKit Voice Agent

A voice AI agent using LiveKit Agents framework with custom F5-TTS and Faster-Whisper plugins.
"""

import logging
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext, JobProcess
from livekit.plugins import silero

from config import (
    logger,
    SYSTEM_PROMPT,
    REF_AUDIO_PATH,
    REF_TEXT,
    SPEED,
    DEVICE,
    STT_DEVICE,
    OLLAMA_HOST,
)
from plugins import F5TTS, FasterWhisperSTT, OllamaLLM

# Load environment variables
load_dotenv()


class SirHenryAgent(Agent):
    """The Sir Henry voice AI agent."""

    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)

    async def on_enter(self):
        """Called when the agent enters a session."""
        # Generate an initial greeting
        self.session.generate_reply(
            instructions="Introduce yourself briefly with a pirate greeting."
        )


def prewarm(proc: JobProcess):
    """
    Prewarm function to load models before the agent starts.
    This reduces latency when the first user connects.
    """
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
        llm=OllamaLLM(
            model="llama3.2:3b",
            host=OLLAMA_HOST,
        ),
    )

    # Start the session with our agent
    await session.start(
        agent=SirHenryAgent(),
        room=ctx.room,
    )

    logger.info("Agent session started.")


def main():
    """Run the LiveKit agent."""
    logger.info("Starting Sir Henry LiveKit Agent...")

    # Create the agent server
    worker = agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    )

    # Run the agent
    agents.cli.run_app(worker)


if __name__ == "__main__":
    main()
