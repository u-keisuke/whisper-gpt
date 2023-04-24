import argparse
import os
import sys

import openai
from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm

# APIキーを設定する
openai.api_key = os.environ["OPENAI_API_KEY"]

parser = argparse.ArgumentParser(description="Transcribe Whisper Audio Files")
parser.add_argument("input_dir", type=str, help="Input directory")
parser.add_argument("output_dir", type=str, help="Output directory")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# 言語コードを指定するオプションを定義する
language_code = "en-US"
language_option = {"language": language_code}

for input_file_name in sorted(os.listdir(args.input_dir)):
    # 入力ファイルのパスを構築する
    input_file_path = os.path.join(args.input_dir, input_file_name)

    # 出力ファイルのパスを構築する
    output_file_name = f"{os.path.splitext(input_file_name)[0]}_transcript.txt"
    output_file_path = os.path.join(args.output_dir, output_file_name)

    print(f"{input_file_name} -> {output_file_name}")
    
    # mp3とmp4で処理を分ける
    if input_file_path.endswith(".mp3"):
        # mp3ファイルを読み込む
        sound = AudioSegment.from_file(input_file_path, format="mp3")
    elif input_file_path.endswith(".mp4"):
        # mp4ファイルを読み込む
        sound = AudioSegment.from_file(input_file_path, format="mp4")

    # ファイルの長さを取得する
    file_length = len(sound)

    # 分割する秒数を指定する
    split_length = 30 * 60 * 1000  # 30 min

    # 分割した音声ファイルを保存するリストを作成する
    chunk_files = []

    # 分割した音声ファイルを生成する
    for i in range(0, file_length, split_length):
        # 分割範囲を指定する
        start = i
        end = min(i + split_length, file_length)
        
        # 分割範囲の音声を取得する
        split_sound = sound[start:end]

        chunk_file = f"chunk_{i // split_length}.mp3"
        chunk_files.append(chunk_file)

        # 分割した音声ファイルを保存する
        # 分割ファイルが存在する場合はスキップする
        if not os.path.exists(chunk_file):
            split_sound.export(chunk_file, format="mp3")

    # 分割された音声ファイルをWhisper APIに送信して、認識結果を取得する
    for i, chunk_file in enumerate(chunk_files):
        print(f"{i} / {len(chunk_files)-1}")

        audio_file = open(chunk_file, "rb")
        response = openai.Audio.transcribe(
            "whisper-1", 
            audio_file, 
            language="en"
        )

        # 中間ファイルの削除
        os.remove(chunk_file)

        # 認識結果を保存する
        with open(output_file_path, "a") as f:
            f.write(response.text)
