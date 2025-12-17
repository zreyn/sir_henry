"""
Test script to verify microphone capture and playback.
This script captures audio from the microphone and plays it back through speakers
to verify the audio pipeline works correctly.
"""

import asyncio
import sounddevice as sd
import numpy as np


def test_audio_loopback():
    """
    Capture audio from microphone and play it back through speakers.
    This creates a simple loopback to verify audio capture works.
    """
    sample_rate = 48000
    channels = 1
    blocksize = 960  # 20ms frames at 48kHz
    dtype = "float32"
    
    print(f"Starting audio loopback test...")
    print(f"Sample rate: {sample_rate}Hz, Channels: {channels}, Blocksize: {blocksize}")
    print("Speak into your microphone - you should hear it through your speakers.")
    print("Press Ctrl+C to stop.\n")
    
    def audio_callback(indata, outdata, frames, time, status):
        """Callback that copies input directly to output (loopback)."""
        if status:
            print(f"Audio status: {status}")
        
        # Copy input directly to output
        outdata[:] = indata
        
        # Calculate RMS level for monitoring
        rms = np.sqrt(np.mean(indata**2))
        if rms > 0.01:  # Only print if there's significant audio
            print(f"Audio level: {rms:.4f} ({20 * np.log10(rms + 1e-10):.1f} dB)", end='\r')
    
    try:
        # Create a duplex stream (input + output)
        with sd.Stream(
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype,
            blocksize=blocksize,
            callback=audio_callback,
        ):
            print("Audio stream started. Listening...")
            # Keep running until interrupted
            while True:
                sd.sleep(1000)  # Sleep in milliseconds
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")


def test_audio_capture_format(gain: float = 3.0):
    """
    Test capturing audio and verify the format matches what we send to LiveKit.
    This captures audio, converts it to int16 format like we do for LiveKit,
    and then converts back to float32 for playback.
    
    Args:
        gain: Audio gain multiplier (default 3.0 = 3x amplification)
    """
    sample_rate = 48000
    channels = 1
    blocksize = 960  # 20ms frames at 48kHz
    
    print(f"Starting audio format test with {gain}x gain...")
    print(f"Sample rate: {sample_rate}Hz, Channels: {channels}, Blocksize: {blocksize}")
    print("This test captures audio, converts to int16 (like LiveKit), then back to float32 for playback.")
    print("Press Ctrl+C to stop.\n")
    
    frame_count = 0
    
    def audio_callback(indata, outdata, frames, time, status):
        nonlocal frame_count
        if status:
            print(f"Audio status: {status}")
        
        try:
            # Apply gain amplification (like in main.py)
            amplified = indata * gain
            amplified = np.clip(amplified, -1.0, 1.0)
            
            # Convert float32 to int16 (like we do for LiveKit)
            audio_int16 = (amplified * 32767).astype("int16").flatten()
            
            # Convert back to float32 for playback
            audio_float32 = (audio_int16.astype("float32") / 32767).reshape(-1, channels)
            
            # Output the converted audio
            outdata[:] = audio_float32
            
            frame_count += 1
            if frame_count % 50 == 0:  # Log every 50 frames (~1 second)
                rms_original = np.sqrt(np.mean(indata**2))
                rms_amplified = np.sqrt(np.mean(amplified**2))
                print(f"Processed {frame_count} frames, Original RMS: {rms_original:.4f}, Amplified RMS: {rms_amplified:.4f}")
        except Exception as e:
            print(f"Error in callback: {e}")
    
    try:
        with sd.Stream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=blocksize,
            callback=audio_callback,
        ):
            print("Audio stream started. Listening...")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    import sys
    
    gain = 40.0
    if len(sys.argv) > 1:
        if sys.argv[1] == "format":
            # Test with format conversion (like LiveKit)
            if len(sys.argv) > 2:
                try:
                    gain = float(sys.argv[2])
                except ValueError:
                    print(f"Invalid gain value: {sys.argv[2]}, using default 3.0")
            test_audio_capture_format(gain=gain)
        else:
            # Simple loopback test
            test_audio_loopback()
    else:
        # Simple loopback test
        test_audio_loopback()

