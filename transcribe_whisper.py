import argparse
import os
import sys
import openai
from pydub import AudioSegment
from tqdm import tqdm


class AudioTranscriber:
    def __init__(self, input_dir, output_dir, language_code="en"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.language_code = language_code
        self.ensure_output_directory_exists()

    def ensure_output_directory_exists(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def transcribe_audio_file(self, input_file_path, output_file_path):
        sound = self.load_audio_file(input_file_path)
        file_length = len(sound)
        split_length = 30 * 60 * 1000  # 30 min
        chunk_files = self.create_chunks(sound, file_length, split_length)

        for i, chunk_file in enumerate(chunk_files):
            print(f"{i+1} / {len(chunk_files)}")
            audio_file = open(chunk_file, "rb")
            response = openai.Audio.transcribe("whisper-1", audio_file, language=self.language_code)
            os.remove(chunk_file)
            self.save_transcription(output_file_path, response.text)

    def load_audio_file(self, input_file_path):
        if input_file_path.endswith(".mp3"):
            return AudioSegment.from_file(input_file_path, format="mp3")
        elif input_file_path.endswith(".mp4"):
            return AudioSegment.from_file(input_file_path, format="mp4")

    def create_chunks(self, sound, file_length, split_length):
        chunk_files = []

        for i in range(0, file_length, split_length):
            start = i
            end = min(i + split_length, file_length)
            split_sound = sound[start:end]
            chunk_file = f"chunk_{i // split_length}.mp3"
            chunk_files.append(chunk_file)

            if not os.path.exists(chunk_file):
                split_sound.export(chunk_file, format="mp3")

        return chunk_files

    def save_transcription(self, output_file_path, text):
        with open(output_file_path, "a") as f:
            f.write(text)

    def transcribe_all_files(self):
        for input_file_name in sorted(os.listdir(self.input_dir)):
            input_file_path = os.path.join(self.input_dir, input_file_name)
            output_file_name = f"{os.path.splitext(input_file_name)[0]}_transcript.txt"
            output_file_path = os.path.join(self.output_dir, output_file_name)
            print(f"{input_file_name} -> {output_file_name}")
            self.transcribe_audio_file(input_file_path, output_file_path)


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    parser = argparse.ArgumentParser(description="Transcribe Whisper Audio Files")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--language_code", type=str, default="en", help="Language code")
    args = parser.parse_args()

    audio_transcriber = AudioTranscriber(args.input_dir, args.output_dir, args.language_code)
    audio_transcriber.transcribe_all_files()


if __name__ == "__main__":
    main()
