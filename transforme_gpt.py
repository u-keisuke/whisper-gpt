import argparse
import json
import os
import sys

import openai


class TextTransformer:
    def __init__(self, input_dir, output_dir, prompt_file, log_base_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_dir = os.path.join(log_base_dir, "gpt")
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
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def transform_text(self, text):
        response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo",
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.prompt_text},
                {"role": "user", "content": text},
            ]
        )
        return response

    def transform_file(self, input_file_path, output_file_path, log_file_path):
        text = open(input_file_path, "r").read()
        chunks = self.split_text_into_chunks(text)

        for i, chunk in enumerate(chunks):
            print(f"{i+1} / {len(chunks)}:", end=" ")

            response = self.transform_text(chunk)

            try:
                prompt_tokens = response["usage"]["prompt_tokens"]
                completion_tokens = response["usage"]["completion_tokens"]
            except KeyError:
                prompt_tokens = "N/A"
                completion_tokens = "N/A"
            
            print(f"tokens: ({prompt_tokens=}, {completion_tokens=})")

            self.save_result(output_file_path, "# Chunk " + str(i+1) + " / " + str(len(chunks)))
            self.save_result(output_file_path, response)
            self.save_log(log_file_path, response)
        
        print()

    def split_text_into_chunks(self, text, chunk_size=2400):
        chunks = []
        input_list = text.split()
        for i in range(0, len(input_list), chunk_size):
            chunk = " ".join(input_list[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def save_result(self, output_file_path, response):
        if type(response) == openai.openai_object.OpenAIObject:
            response_text = response["choices"][0]["message"]["content"] + "\n"
        elif type(response) == str:
            response_text = response
        else:
            print("ERROR: response is not a dict or string")
            sys.exit(1)
        with open(output_file_path, "a") as f:
            f.write(response_text + "\n\n")

    def save_log(self, log_file_path, response):
        with open(log_file_path, "a") as f:
            json.dump(response, f, indent=4)
            f.write("\n")

    def transform_all_files(self):
        for input_file_name in sorted(os.listdir(self.input_dir)):
            # exclude system files
            if input_file_name.startswith("."):
                continue
                
            input_file_path = os.path.join(self.input_dir, input_file_name)

            output_file_name = input_file_name.replace("_transcript", f"_{self.prompt_name}")
            output_file_path = os.path.join(self.output_dir, output_file_name)

            log_file_name = input_file_name.replace("_transcript.txt", f"_{self.prompt_name}_log.jsonl")
            log_file_path = os.path.join(self.log_dir, log_file_name)

            print(f"{input_file_name} -> {output_file_name}")
            self.transform_file(input_file_path, output_file_path, log_file_path)


def main():
    # set API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    parser = argparse.ArgumentParser(description="Transform text files using GPT-3")
    parser.add_argument("input_dir", type=str, help="Input directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("prompt_file", type=str, help="Prompt file path")
    parser.add_argument("--log_dir", type=str, default="log", help="Log directory")
    args = parser.parse_args()

    texttransformer = TextTransformer(args.input_dir, args.output_dir, args.prompt_file, args.log_dir)
    texttransformer.transform_all_files()


if __name__ == "__main__":
    main()
