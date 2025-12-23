#!/usr/bin/env python3
"""
Interactive F5-TTS tester.

Test the F5-TTS service with different characters and measure latency.
"""

import asyncio
import time
import numpy as np

import aiohttp

from src.config import CHARACTERS, TTS_HOST


def get_f5_characters() -> dict:
    """Return only characters that use F5 TTS."""
    return {
        name: config
        for name, config in CHARACTERS.items()
        if config.get("tts_type", "f5").lower() == "f5"
    }


def select_character() -> tuple[str, dict]:
    """Present character selection menu and return selected character."""
    f5_chars = get_f5_characters()
    char_list = list(f5_chars.items())

    print("\n" + "=" * 50)
    print("F5-TTS Character Tester")
    print("=" * 50)
    print("\nAvailable characters:\n")

    for i, (name, config) in enumerate(char_list, 1):
        greeting = config.get("greeting", "")[:50]
        print(f"  {i}. {name}")
        print(f"     \"{greeting}...\"")
        print()

    while True:
        try:
            choice = input("Select a character (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(char_list):
                name, config = char_list[idx]
                print(f"\n✓ Selected: {name}\n")
                return name, config
            else:
                print(f"Please enter a number between 1 and {len(char_list)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)


async def synthesize_audio(
    session: aiohttp.ClientSession,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    speed: float,
    service_url: str,
) -> tuple[bytes, float]:
    """
    Synthesize audio using F5-TTS service.
    Returns (audio_bytes, latency_ms).
    """
    payload = {
        "text": text,
        "ref_audio_path": ref_audio_path,
        "ref_text": ref_text,
        "speed": speed,
    }

    start_time = time.perf_counter()

    async with session.post(f"{service_url}/synthesize", json=payload) as resp:
        if resp.status == 200:
            audio_data = await resp.read()
            latency_ms = (time.perf_counter() - start_time) * 1000
            return audio_data, latency_ms
        else:
            error = await resp.text()
            raise RuntimeError(f"TTS service error ({resp.status}): {error}")


def play_audio(audio_bytes: bytes, sample_rate: int = 24000):
    """Play PCM audio bytes."""
    try:
        import sounddevice as sd

        # Convert bytes to numpy array (int16 PCM)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        # Normalize to float32 for playback
        audio_float = audio.astype(np.float32) / 32768.0

        sd.play(audio_float, sample_rate)
        sd.wait()
    except ImportError:
        print("  [sounddevice not installed - audio playback disabled]")
        print("  Install with: pip install sounddevice")


async def interactive_loop(char_name: str, char_config: dict, service_url: str):
    """Main interactive loop for TTS testing."""
    ref_audio_path = char_config.get("ref_audio_path", "")
    ref_text = char_config.get("ref_text", "")
    speed = char_config.get("speed", 1.0)

    print("=" * 50)
    print(f"Testing: {char_name}")
    print(f"Service: {service_url}")
    print(f"Speed: {speed}")
    print("=" * 50)
    print("\nEnter text to synthesize (or 'quit' to exit):\n")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                text = input("> ").strip()

                if text.lower() in ("quit", "exit", "q"):
                    print("\nGoodbye!")
                    break

                if not text:
                    continue

                print(f"  Synthesizing {len(text)} characters...")

                try:
                    audio_bytes, latency_ms = await synthesize_audio(
                        session=session,
                        text=text,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                        speed=speed,
                        service_url=service_url,
                    )

                    audio_duration_ms = len(audio_bytes) / 2 / 24000 * 1000  # int16 = 2 bytes
                    rtf = latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0

                    print(f"  ✓ Latency: {latency_ms:.0f}ms")
                    print(f"  ✓ Audio duration: {audio_duration_ms:.0f}ms")
                    print(f"  ✓ Real-time factor: {rtf:.2f}x")
                    print("  Playing audio...")

                    play_audio(audio_bytes)
                    print()

                except Exception as e:
                    print(f"  ✗ Error: {e}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break


def main():
    """Main entry point."""
    # Select character
    char_name, char_config = select_character()

    # Run interactive loop
    asyncio.run(interactive_loop(char_name, char_config, TTS_HOST))


if __name__ == "__main__":
    main()
