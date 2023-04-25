import argparse
import json
import os
import sys
import openai


class TextTransformer:
    def __init__(self, input_dir, output_dir, prompt_file):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.prompt_file = prompt_file
        self.prompt_text = self.read_prompt_file()
        prompt_file_name = os.path.splitext(os.path.basename(prompt_file))[0]
        start_position = prompt_file_name.find('-') + 1
        self.prompt_name = prompt_file_name[start_position:]
        self.setup_output_directories()

    def read_prompt_file(self):
        with open(self.prompt_file, "r") as f:
            return f.read()

    def setup_output_directories(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(os.path.join(self.output_dir, "log")):
            os.makedirs(os.path.join(self.output_dir, "log"))

    def transform_file(self, input_file_name):
        input_file_path = os.path.join(self.input_dir, input_file_name)
        output_file_name = input_file_name.replace("_transcript", f"_{self.prompt_name}")
        output_file_path = os.path.join(self.output_dir, output_file_name)
        log_file_name = input_file_name.replace("_transcript.txt", f"_{self.prompt_name}_log.json")
        log_file_path = os.path.join(self.output_dir, "log", log_file_name)

        print(f"{input_file_name} -> {output_file_name}")

        text = open(input_file_path, "r").read()
        chunks = self.split_text_into_chunks(text)

        for i, chunk in enumerate(chunks):
            print(f"{i+1} / {len(chunks)}")

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.prompt_text},
                    {"role": "user", "content": chunk},
                ]
            )

            self.save_result(output_file_path, response)
            self.save_log(log_file_path, response)

    def split_text_into_chunks(self, text, chunk_size=2400):
        chunks = []
        input_list = text.split()
        for i in range(0, len(input_list), chunk_size):
            chunk = " ".join(input_list[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def save_result(self, output_file_path, response):
        with open(output_file_path, "a") as f:
            f.write(response["choices"][0]["message"]["content"]+"\n\n\n")

    def save_log(self, log_file_path, response):
        with open(log_file_path, "a") as f:
            json.dump(response, f, indent=4)
            f.write("\n")

    def transform_all_files(self):
        for input_file_name in sorted(os.listdir(self.input_dir)):
            self.transform_file(input_file_name)


def main():
    # set API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    parser = argparse.ArgumentParser(description="Transform text files using GPT-3")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("prompt_file", type=str, help="Prompt file path")
    args = parser.parse_args()

    texttransformer = TextTransformer(args.input_dir, args.output_dir, args.prompt_file)
    texttransformer.transform_all_files()


if __name__ == "__main__":
    main()
