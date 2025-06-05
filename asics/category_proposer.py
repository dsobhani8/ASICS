#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from pathlib import Path
import openai
import re


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


def debug_process_chunk(chunk_text: str):
    """
    Print out the chunk text, the regex pattern, and the findall result.
    """
    print("===== RAW CHUNK BEGINS =====")
    # Make newlines visible by replacing them with a literal ‘⏎\n’ marker
    visible = chunk_text.replace("\n", "\\n\n")
    print(visible)
    print("===== RAW CHUNK ENDS =====\n")

    # The regex we’re trying to use:
    pattern = (
        r"\d+\.\s"                                # “1. ”, “2. ”, etc.
        r"\*\*.*?Title\*\*:\s*(.*?)\n"            # capture Task Title text up to newline
        r"\*\*.*?Description\*\*:\s*"
        r"(.*?)(?=\n\s*-\s|\Z)"                  # capture Task Description until “- ” or EOS
    )
    print("Regex pattern:")
    print(pattern + "\n")

    # Show what re.findall returns
    matches = re.findall(pattern, chunk_text, flags=re.DOTALL)
    print(f"re.findall found {len(matches)} match(es):")
    for i, (title, desc) in enumerate(matches, start=1):
        print(f" Match #{i}:")
        print("  Title raw →", repr(title[:50] + ("…" if len(title)>50 else "")))
        print("  Desc raw  →", repr(desc[:50] + ("…" if len(desc)>50 else "")))
    print("\n--- End of debug_process_chunk ---\n")

def process_chunk(chunk_text: str) -> pd.DataFrame:
    # (Call debug_process_chunk if you want to inspect why it isn’t matching)
    debug_process_chunk(chunk_text)
    print('here')
    rows = []
    pattern = (
        r"\d+\.\s"
        r"\*\*.*?Title\*\*:\s*(.*?)\n"
        r"\*\*.*?Description\*\*:\s*"
        r"(.*?)(?=\n\s*-\s|\Z)"
    )
    matches = re.findall(pattern, chunk_text, flags=re.DOTALL)
    for title_text, desc_text in matches:
        title_text = title_text.strip()
        desc_text = desc_text.strip()
        rows.append({
            "Task Title": title_text,
            "Task Description": desc_text
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Load a dataset and call OpenAI API in chunks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="hpd",
        help=(
            "Name of the dataset (without “.csv”). "
            "Default: “hpd” → loads “hpd_llama_vs_claude.csv”."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Number of `system_input` rows to send per API call. Default: 60.",
    )
    parser.add_argument(
        "--percentage",        # ← new argument
        type=float,
        default=100.0,
        help=(
            "Percentage of the dataset to process (0–100). "
            "Default: 100 (process all rows)."
        ),
    )
    args = parser.parse_args()

    # 1) Read OpenAI settings from env
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model_name = os.getenv("MODEL_NAME", "")
    if not model_name:
        raise RuntimeError("MODEL_NAME is not set")

    # 2) Load the full DataFrame
    try:
        df = load_dataset(args.dataset)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading dataset: {e}")
        return

    # 3) Extract and clean the “system_input” column
    all_inputs = df["system_input"].dropna().astype(str).tolist()

    # 4) Apply percentage slicing (so we only keep that fraction of rows)  ← added percentage logic
    if not (0.0 < args.percentage <= 100.0):
        raise ValueError("--percentage must be > 0 and ≤ 100")
    total_rows = len(all_inputs)
    num_to_process = int(total_rows * (args.percentage / 100.0))
    # If percentage = 100, num_to_process == total_rows
    inputs = all_inputs[:num_to_process]

    # 5) Load the prompt template from prompts/category_proposer.txt
    try:
        prompt_template = load_prompt_template()
    except FileNotFoundError as e:
        print(f"Error loading prompt template: {e}")
        return

    response_dict = {}

    # 6) Iterate over chunks of `inputs` and call the OpenAI API
    for chunk_idx, chunk in enumerate(chunked_iterable(inputs, args.chunk_size), start=1):
        joined_inputs = "\n\n".join(chunk)
        prompt = prompt_template.replace("{system inputs}", joined_inputs)

        try:
            response = openai.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=[
                {"role": "system", "content": "You are an expert at identifying categories in questions."},
                {"role": "user", "content": prompt}
                ]
            )

        except Exception as api_err:
            print(f"[Chunk {chunk_idx}] OpenAI API error: {api_err}")
            continue

        reply = response.choices[0].message.content
        print(f"\n--- Chunk {chunk_idx} Response ---\n{reply}\n")
        response_dict[chunk_idx] = reply


    df_list = []
    for chunk_idx, raw_text in response_dict.items():
        df_chunk = process_chunk(raw_text)
        print(raw_text)
        print('----')
        print(df_chunk)
        # Optionally keep track of which chunk this row came from:
        df_chunk["chunk_index"] = chunk_idx
        df_list.append(df_chunk)

    # 2) Concatenate all per‐chunk DataFrames (if any)
    if df_list:
        all_tasks_df = pd.concat(df_list, ignore_index=True)
    else:
        # No data extracted—create an empty DataFrame with the right columns
        all_tasks_df = pd.DataFrame(columns=["Task Title", "Task Description", "chunk_index"])

    # 3) Save to CSV (or JSON)
    all_tasks_df.to_csv("proposed_tasks.csv", index=False)
    # Or, if you prefer JSON:
    # all_tasks_df.to_json("proposed_tasks.json", orient="records", lines=False)

    print(f"Saved {len(all_tasks_df)} total tasks to 'proposed_tasks.csv'")
    print(all_tasks_df.head())
if __name__ == "__main__":
    main()