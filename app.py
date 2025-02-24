import argparse
import os
import time
from typing import Tuple, List

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def correct_spelling(text_to_correct: str, instruction: str, temperature: float) -> str:
    # fix multiple whitespaces
    text_to_correct = " ".join(text_to_correct.split())
    response = openai.Edit.create(
        engine="text-davinci-edit-001",
        input=text_to_correct,
        instruction=instruction,
        temperature=temperature
    )
    correction = " ".join(response["choices"][0]["text"].split())
    return correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default=None)
    parser.add_argument("--out-file", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--instruction", type=str, default="Fix the spelling mistakes")
    parser.add_argument("--temperature", type=float, default=0)
    return parser.parse_args()


def load_text_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf8") as inf:
        lines = [line.strip() for line in inf]
    return lines


def line_count(path: str) -> int:
    num_lines = 0
    with open(path, "r", encoding="utf8") as inf:
        for _ in inf:
            num_lines += 1
    return num_lines


def run(args: argparse.Namespace) -> None:
    if args.text is not None:
        start = time.perf_counter()
        correction = correct_spelling(args.text, args.instruction, args.temperature)
        runtime = time.perf_counter() - start
        print(f"Input text:\t{args.text}\nCorrection:\t{correction}\nRuntime:\t{runtime:.2f}s")
        return

    assert args.in_file is not None and args.out_file is not None, "in-file and out-file arguments must be specified"
    if os.path.exists(args.out_file) and not args.overwrite and not args.resume:
        print(f"Out file at {args.out_file} already exists")
        return

    start_overall = time.perf_counter()
    inputs = load_text_file(args.in_file)
    print(f"Got {len(inputs)} inputs to correct")
    already_processed = line_count(args.out_file) if os.path.exists(args.out_file) and args.resume else 0
    if args.resume:
        print(f"Resuming correcting file {args.in_file}, already got {already_processed} corrections, "
              f"{len(inputs) - already_processed} left.")
    inputs = inputs[already_processed:]
    total_runtime = 0
    with open(args.out_file, "a" if args.resume else "w", encoding="utf8", buffering=1) as of:
        for i, ipt in enumerate(inputs):
            while True:
                start = time.perf_counter()
                try:
                    correction = correct_spelling(ipt, args.instruction, args.temperature)
                except openai.error.RateLimitError:
                    print("Hit rate limit, trying again in 5s")
                    time.sleep(5)
                    continue
                except openai.error.OpenAIError as e:
                    print(f"OpenAI exception while correcting text '{ipt.strip()}':")
                    if str(e).startswith("Could not edit text."):
                        print(f"It seems as if OpenAI could not edit the text with the current temperature setting "
                              f"of t={args.temperature}, keeping the input unchanged.")
                        correction = ipt
                    else:
                        print(e)
                        return

                of.write(" ".join(correction.split()) + "\n")
                end = time.perf_counter()
                total_runtime += end - start
                break
            print(f"Progress: {i + 1}/{len(inputs)}")

    end_overall = time.perf_counter()
    print(
        f"Finished correcting {len(inputs)} inputs\n"
        f"Total runtime: {total_runtime:.2f}s ({total_runtime / len(inputs):.2f}s/seq)\n"
        f"Total runtime + waiting because of rate limiting: {end_overall - start_overall:.2f}s"
    )


if __name__ == "__main__":
    run(parse_args())
