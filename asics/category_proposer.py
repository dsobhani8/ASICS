#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from pathlib import Path
import openai

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Given a dataset name, load the corresponding CSV from ~/ASICS/data/.
    Defaults:
      - If dataset_name == "hpd", load “hpd_llama_vs_claude.csv”.
      - Otherwise, load “{dataset_name}.csv”.
    Raises:
      FileNotFoundError if the CSV does not exist.
    """
    data_dir = Path(os.path.expanduser("~/ASICS/data"))
    if dataset_name.lower() == "hpd":
        filename = "hpd_llama_vs_claude.csv"
    else:
        filename = f"{dataset_name}.csv"

    dataset_path = data_dir / filename
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Could not find {dataset_path}")

    df = pd.read_csv(dataset_path)
    # Ensure the required column “system_input” exists
    if "system_input" not in df.columns:
        raise KeyError(f"'system_input' column not found in {dataset_path}")
    return df

def load_prompt_template() -> str:
    """
    Starting from the directory of this script, walk upward until you find
    a “prompts/category_proposer.txt” under an ASICS folder. Return its contents.
    """
    # 1) Start in the directory where this script lives
    current = Path(__file__).resolve().parent

    # 2) Keep going up until the filesystem root
    while True:
        candidate = current / "prompts" / "category_proposer.txt"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

        # If we reach the top (parent == self), stop searching
        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError(
        "Could not locate 'prompts/category_proposer.txt' anywhere above "
        f"the script’s path ({Path(__file__).resolve()})."
    )

def chunked_iterable(iterable, size):
    """Yield successive chunks of `size` from `iterable` (any sequence)."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

def main():
    parser = argparse.ArgumentParser(
        description="Load a dataset and call OpenAI API in chunks to propose categories."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hpd",
        help="Name of the dataset (without “.csv”). Default: “hpd” → loads “hpd_llama_vs_claude.csv”.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Number of `system_input` rows to send per API call. Default: 60.",
    )
    args = parser.parse_args()

    # 1) Read environment variables for OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("Environment variable OPENAI_API_KEY is not set.")

    model_name = os.getenv("MODEL_NAME", "")
    if not model_name:
        raise RuntimeError("Environment variable MODEL_NAME is not set.")

    # 2) Load dataset
    try:
        df = load_dataset(args.dataset)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading dataset: {e}")
        return

    # 3) Load category proposer prompt template
    try:
        prompt_template = load_prompt_template()
    except FileNotFoundError as e:
        print(f"Error loading prompt template: {e}")
        return

    # 4) Extract the “system_input” column, drop any NaNs
    inputs = df["system_input"].dropna().astype(str).tolist()

    # 5) For each chunk of size N, substitute {system inputs} and call the OpenAI API
    for chunk_idx, chunk in enumerate(chunked_iterable(inputs, args.chunk_size), start=1):
        # Join the chunk into a single string, each input separated by two newlines
        joined_inputs = "\n\n".join(chunk)

        # Insert into the prompt
        prompt = prompt_template.replace("{system inputs}", joined_inputs)

        # Call ChatCompletion (assuming a chat‐based model)
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                {"role": "system", "content": "You are an expert at extracting text from articles exactly without adding or deleting."},
                {"role": "user", "content": prompt}
                ]
            )

        except Exception as api_err:
            print(f"[Chunk {chunk_idx}] OpenAI API error: {api_err}")
            continue

        # Extract the assistant’s reply
        content = response.choices[0].message.content.strip()
        print(f"\n--- Chunk {chunk_idx} Response ---\n{content}\n")

if __name__ == "__main__":
    main()
