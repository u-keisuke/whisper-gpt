import argparse
import json
import os
import sys

import openai

# APIキーを設定する
openai.api_key = os.environ["OPENAI_API_KEY"]

parser = argparse.ArgumentParser(description="Transcribe Whisper Audio Files")
parser.add_argument("input_dir", type=str, help="Input directory")
parser.add_argument("output_dir", type=str, help="Output directory")
parser.add_argument("prompt_file", type=str, help="Prompt file path")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
if not os.path.exists(os.path.join(args.output_dir, "log")):
    os.makedirs(os.path.join(args.output_dir, "log"))

# プロンプトファイルの内容を読み込む
with open(args.prompt_file, "r") as f:
    prompt_text = f.read()

for input_file_name in sorted(os.listdir(args.input_dir)):
    # 入力ファイルのパスを構築する
    input_file_path = os.path.join(args.input_dir, input_file_name)

    output_file_name = input_file_name.replace("_transcript", "_summary")
    output_file_path = os.path.join(args.output_dir, output_file_name)
    log_file_name = input_file_name.replace("_transcript.txt", "_gptlog.json")
    log_file_path = os.path.join(args.output_dir, "log", log_file_name)

    print(f"{input_file_name} -> {output_file_name}")

    # 入力テキストファイルを読み込む
    text = open(input_file_path, "r").read()

    # テキストをピリオドの後で分割する
    sentences = text.split(".")
    chunks = []
    chunk = ""

    for sentence in sentences:
        # 文章が最大トークン数を超える場合は、前の文章までを1つのチャンクとする
        if len((chunk + sentence).split()) > 2800:
            chunks.append(chunk)
            chunk = ""

        # ピリオドの後で文章を分割
        if chunk == "":
            chunk += sentence.strip() + "."
        else:
            chunk += " " + sentence.strip() + "."

    # 最後のチャンクを追加
    if chunk:
        chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        print(f"{i} / {len(chunks)-1}")
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                #{"role": "system", "content": "あなたは役に立つアシスタントです。"},
                #{"role": "user", "content": "2021年の日本シリーズで優勝したのは?"},
                #{"role": "assistant", "content": "2021年の日本シリーズで優勝したのは、東京ヤクルトスワローズです。"},
                #{"role": "user", "content": "その球団の本拠地はどこですか?"}
                {
                    "role": "system", 
                    "content": prompt_text
                },
                {"role": "user", "content": chunk},
            ]
        )
        
        # 認識結果を保存
        with open(output_file_path, "a") as f:
            f.write(response["choices"][0]["message"]["content"]+"\n\n\n")
        
        # レスポンスをJSON形式で保存
        with open(log_file_path, "a") as f:
            json.dump(response, f, indent=4)
            f.write("\n")