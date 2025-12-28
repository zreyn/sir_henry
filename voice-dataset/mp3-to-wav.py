from pydub import AudioSegment
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("elevenlabs-client")


def main():
    src_folder = "raw/mp3"
    dst_folder = "raw/wav"

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    audio_files = os.listdir(src_folder)

    for file in audio_files:
        if file.endswith(".mp3"):
            name, ext = os.path.splitext(file)
            output_path = os.path.join(dst_folder, f"{name}.wav")
            logger.info(f"Converting {file} to {output_path}")

            # try:
            #     # Load and export the file
            #     sound = AudioSegment.from_mp3(file)
            #     sound.export(output_path, format="wav")
            #     print(f"Converted: {file} -> {os.path.basename(output_path)}")
            # except Exception as e:
            #     print(f"Failed to convert {file}: {e}")


if __name__ == "__main__":
    main()
