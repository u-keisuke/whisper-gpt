import argparse
import json
import os
import sys

import openai
from pydub import AudioSegment
from tqdm import tqdm


class AudioTranscriber:
    def __init__(self, input_dir, output_dir, log_base_dir, language_code="en"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_dir = os.path.join(log_base_dir, "whisper")
        self.language_code = language_code
        self.setup_output_directories()

    def setup_output_directories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def transcribe_audio(self, audio_data):
        response = openai.Audio.transcribe("whisper-1", audio_data, language=self.language_code)
        return response

    def transcribe_file(self, input_file_path, output_file_path, log_file_path):
        sound = self.load_audio_file(input_file_path)
        chunk_files = self.split_audio_into_chunks(sound)

        for i, chunk_file in enumerate(chunk_files):
            print(f"{i+1} / {len(chunk_files)}")

            audio_data = open(chunk_file, "rb")
            response = self.transcribe_audio(audio_data)
            audio_data.close()

            self.save_result(output_file_path, response.text)
            self.save_log(log_file_path, response)
            os.remove(chunk_file)

        print()

    def load_audio_file(self, input_file_path):
        if input_file_path.endswith(".mp3"):
            return AudioSegment.from_file(input_file_path, format="mp3")
        elif input_file_path.endswith(".mp4"):
            return AudioSegment.from_file(input_file_path, format="mp4")

    def split_audio_into_chunks(self, sound, chunk_length=30*60*1000): # 30 min split
        chunk_files = []
        file_length = len(sound)

        for i in range(0, file_length, chunk_length):
            start = i
            end = min(i + chunk_length, file_length)
            split_sound = sound[start:end]
            chunk_file = f"tmp_chunk_{i // chunk_length}.mp3"
            chunk_files.append(chunk_file)

            split_sound.export(chunk_file, format="mp3")

        return chunk_files

    def save_result(self, output_file_path, text):
        with open(output_file_path, "a") as f:
            f.write(text)

    def save_log(self, log_file_path, response):
        with open(log_file_path, "a") as f:
            json.dump(response, f, indent=4)
            f.write("\n")

    def transcribe_all_files(self):
        for input_file_name in sorted(os.listdir(self.input_dir)):
            # exclude system files
            if input_file_name.startswith("."):
                continue

            input_file_path = os.path.join(self.input_dir, input_file_name)

            output_file_name = f"{os.path.splitext(input_file_name)[0]}_transcript.txt"
            output_file_path = os.path.join(self.output_dir, output_file_name)

            log_file_name = f"{os.path.splitext(input_file_name)[0]}_transcript_log.jsonl"
            log_file_path = os.path.join(self.log_dir, log_file_name)

            print(f"{input_file_name} -> {output_file_name}")
            self.transcribe_file(input_file_path, output_file_path, log_file_path)


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    parser = argparse.ArgumentParser(description="Transcribe Whisper Audio Files")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--log_dir", type=str, default="log", help="Log directory")
    parser.add_argument("--language_code", type=str, default="en", help="Language code")
    args = parser.parse_args()

    audio_transcriber = AudioTranscriber(args.input_dir, args.output_dir, args.log_dir, args.language_code)
    audio_transcriber.transcribe_all_files()


if __name__ == "__main__":
    main()
