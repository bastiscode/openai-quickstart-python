import argparse
import os
import time
from typing import Tuple, List

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def correct_spelling(text_to_correct: str) -> Tuple[str, float]:
    # fix multiple whitespaces
    start = time.perf_counter()
    text_to_correct = " ".join(text_to_correct.split())
    response = openai.Edit.create(
        engine="text-davinci-edit-001",
        input=text_to_correct,
        instruction="Fix the spelling mistakes",
        temperature=0.
    )
    end = time.perf_counter()
    correction = " ".join(response["choices"][0]["text"].split())
    return correction, end - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default=None)
    parser.add_argument("--out-file", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_text_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf8") as inf:
        lines = [line.strip() for line in inf]
    return lines


def save_text_file(path: str, contents: List[str]) -> None:
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as of:
        for content in contents:
            of.write(content + "\n")


def run(args: argparse.Namespace) -> None:
    if args.text is not None:
        correction, runtime = correct_spelling(args.text)
        print(f"Input text:\t{args.text}\nCorrection:\t{correction}\nRuntime:\t{runtime:.2f}s")
        return

    assert args.in_file is not None and args.out_file is not None, "in-file and out-file arguments must be specified"
    if os.path.exists(args.out_file) and not args.overwrite:
        print(f"Out file at {args.out_file} already exists")
        return

    inputs = load_text_file(args.in_file)
    print(f"Got {len(inputs)} inputs to correct")
    outputs = []
    total_runtime = 0
    for i, ipt in enumerate(inputs):
        while True:
            try:
                correction, runtime = correct_spelling(ipt)
            except openai.error.RateLimitError:
                print("Hit rate limit, trying again in 5s")
                time.sleep(5)
                continue
            except openai.error.OpenAIError as e:
                print(f"OpenAI exception: {e}")
                return

            outputs.append(correction)
            total_runtime += runtime
            break
        print(f"Progress: {i + 1}/{len(inputs)}")
    print(f"Finished correcting {len(inputs)} inputs. "
          f"Total runtime: {total_runtime:.2f}s ({total_runtime / len(inputs):.2f}s/seq)")
    save_text_file(args.out_file, outputs)


if __name__ == "__main__":
    run(parse_args())
