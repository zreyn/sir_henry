import asyncio
import logging
import os

from livekit import rtc
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

URL = os.environ.get("LIVEKIT_URL")
TOKEN = os.environ.get("LIVEKIT_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sir_henry")


async def _play_audio_stream(audio_stream: rtc.AudioStream) -> None:
    """
    Consume incoming PCM frames from LiveKit and play them through the default
    system output using sounddevice.
    """
    stream = sd.RawOutputStream(
        samplerate=48000,
        channels=1,
        dtype="int16",
        blocksize=960,  # 20ms frames at 48kHz
    )
    stream.start()
    try:
        async for frame_event in audio_stream:
            # AudioStream yields AudioFrameEvent objects, which have a .frame attribute
            audio_frame = frame_event.frame
            # AudioFrame.data is a memoryview of int16 data
            data = bytes(audio_frame.data)
            if data:
                stream.write(data)
    except Exception as e:
        logger.info(f"Error playing audio stream: {e}")
    finally:
        stream.stop()
        stream.close()


async def _capture_microphone(audio_source: rtc.AudioSource, gain: float = 3.0) -> None:
    """
    Capture microphone audio and push it to the AudioSource.
    
    Args:
        audio_source: The LiveKit AudioSource to push frames to
        gain: Audio gain multiplier (default 3.0 = 3x amplification)
    """
    sample_rate = 48000
    channels = 1
    blocksize = 960  # 20ms frames at 48kHz
    
    # Get the event loop for scheduling coroutines from the callback
    loop = asyncio.get_event_loop()
    
    frame_count = 0
    frames_scheduled = 0
    pending_futures = []
    
    def check_futures():
        """Check pending futures for completion and errors."""
        nonlocal pending_futures
        completed = []
        for future in pending_futures:
            if future.done():
                try:
                    future.result()  # This will raise if there was an error
                except Exception as e:
                    logger.info(f"Error in capture_frame future: {e}")
                completed.append(future)
        # Remove completed futures
        pending_futures = [f for f in pending_futures if f not in completed]
        # Keep only last 100 futures to avoid memory issues
        if len(pending_futures) > 100:
            pending_futures = pending_futures[-100:]
    
    def audio_callback(indata, frames, time, status):
        nonlocal frame_count, frames_scheduled
        if status:
            logger.info(f"Audio callback status: {status}")
        try:
            # Calculate RMS to check audio level
            rms = np.sqrt(np.mean(indata**2))
            
            # Apply gain amplification
            amplified = indata * gain
            
            # Clip to prevent distortion (keep within [-1.0, 1.0] range)
            amplified = np.clip(amplified, -1.0, 1.0)
            
            # Convert float32 to int16
            # indata shape is (frames, channels), flatten for mono
            audio_int16 = (amplified * 32767).astype("int16").flatten()
            
            # Push to audio source
            frame = rtc.AudioFrame(
                data=audio_int16.tobytes(),
                sample_rate=sample_rate,
                num_channels=channels,
                samples_per_channel=frames,
            )
            # Schedule the coroutine from the callback thread
            # Note: We can't block here, so we just schedule it and assume it will complete
            # The AudioSource will queue the frame internally
            try:
                future = asyncio.run_coroutine_threadsafe(audio_source.capture_frame(frame), loop)
                frame_count += 1
                frames_scheduled += 1
                pending_futures.append(future)
                
                # Periodically check for completed futures and errors
                if frame_count % 10 == 0:
                    check_futures()
                
                # Log every 50 frames (~1 second) with audio level
                if frame_count % 50 == 0:
                    check_futures()  # Final check before logging
                    completed_count = frames_scheduled - len(pending_futures)
                    logger.info(f"Audio: {frame_count} frames captured, {frames_scheduled} scheduled, {completed_count} completed, {len(pending_futures)} pending, RMS: {rms:.4f} (amplified: {np.sqrt(np.mean(amplified**2)):.4f})")
            except Exception as e:
                logger.info(f"Error scheduling capture_frame: {e}")
        except Exception as e:
            logger.info(f"Error in audio callback: {e}")
    
    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    )
    logger.info(f"Starting microphone capture: {sample_rate}Hz, {channels} channel(s), blocksize={blocksize}")
    stream.start()
    try:
        # Keep running until the audio source is closed
        while True:
            await asyncio.sleep(0.1)
    finally:
        stream.stop()
        stream.close()


async def main():
    room = rtc.Room()

    # 1. Handle Incoming Audio (Speakers)
    @room.on("track_published")
    def on_track_published(publication, participant):
        logger.info(f"Track published by {participant.identity}: {publication.sid} ({publication.kind})")
        logger.info(f"  - Participant SID: {participant.sid}")
        logger.info(f"  - Is local participant: {participant == room.local_participant}")
    
    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        logger.info(f"Track subscribed: {publication.sid} from {participant.identity} ({track.kind})")
        logger.info(f"  - Participant SID: {participant.sid}")
        logger.info(f"  - Is local participant: {participant == room.local_participant}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to {participant.identity}'s audio")
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(_play_audio_stream(audio_stream))
    
    @room.on("participant_connected")
    def on_participant_connected(participant):
        logger.info(f"Participant connected: {participant.identity} (SID: {participant.sid})")
        logger.info(f"  - Is local: {participant == room.local_participant}")
    
    @room.on("track_unsubscribed")
    def on_track_unsubscribed(track, publication, participant):
        logger.info(f"Track unsubscribed: {publication.sid} from {participant.identity}")

    # 2. Connect
    logger.info(f"Connecting to {URL}...")
    await room.connect(URL, TOKEN)
    logger.info("Connected!")

    # 3. Publish Microphone
    audio_source = rtc.AudioSource(48000, 1)  # 48kHz, mono
    local_audio_track = rtc.LocalAudioTrack.create_audio_track("mic_track", audio_source)
    logger.info(f"Created local audio track: {local_audio_track.sid}")
    
    # Check if track is already published
    existing_publications = [p for p in room.local_participant.track_publications.values() 
                            if p.track == local_audio_track]
    if existing_publications:
        logger.info(f"WARNING: Track already has {len(existing_publications)} publication(s)!")
        for pub in existing_publications:
            logger.info(f"  - Existing publication: {pub.sid}")
    
    # Publish with SOURCE_MICROPHONE so the agent recognizes it as a microphone track
    publish_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE,
    )
    publication = await room.local_participant.publish_track(local_audio_track, publish_options)
    logger.info(f"Microphone track published: {publication.sid} (source: MICROPHONE)")
    logger.info(f"Total publications for local participant: {len(room.local_participant.track_publications)}")
    
    # List all participants and their tracks
    logger.info("\nCurrent participants in room:")
    for participant in room.remote_participants.values():
        logger.info(f"  - {participant.identity} (SID: {participant.sid})")
        logger.info(f"    Published tracks: {len(participant.track_publications)}")
        for pub in participant.track_publications.values():
            logger.info(f"      - {pub.sid}: {pub.kind}")
    logger.info(f"  - {room.local_participant.identity} (local, SID: {room.local_participant.sid})")
    logger.info(f"    Published tracks: {len(room.local_participant.track_publications)}")
    for pub in room.local_participant.track_publications.values():
        logger.info(f"      - {pub.sid}: {pub.kind}")
    logger.info("") 
    
    # Start capturing microphone in background
    # Gain of 3.0 amplifies the audio 3x (adjust as needed)
    gain = 40.0
    asyncio.create_task(_capture_microphone(audio_source, gain=gain))
    logger.info(f"Microphone capture started with {gain}x gain.")

    # Keep alive and monitor
    try:
        check_count = 0
        while True:
            await asyncio.sleep(5)
            check_count += 1
            
            # Periodically check if agent has subscribed to our track
            for participant in room.remote_participants.values():
                if "agent" in participant.identity.lower():
                    logger.info(f"[Check {check_count}] Agent {participant.identity} status:")
                    logger.info(f"  - Published tracks: {len(participant.track_publications)}")
                    # Check if agent has subscribed to our track
                    for pub in room.local_participant.track_publications.values():
                        # Check subscribers on the publication
                        subscriber_count = len(pub.subscribers) if hasattr(pub, 'subscribers') else 0
                        logger.info(f"  - Our track {pub.sid} has {subscriber_count} subscriber(s)")
                        if subscriber_count > 0:
                            for sub in pub.subscribers:
                                logger.info(f"    - Subscriber: {sub.identity if hasattr(sub, 'identity') else sub}")
                        else:
                            logger.info(f"  - WARNING: Agent has NOT subscribed to our track {pub.sid}")
                    
                    # Also check what tracks the agent is subscribed to
                    logger.info(f"  - Agent subscribed tracks: {len(participant.track_subscriptions) if hasattr(participant, 'track_subscriptions') else 'N/A'}")
    except KeyboardInterrupt:
        await audio_source.aclose()
        await room.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
