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


def process_chunk(chunk_text: str) -> pd.DataFrame:
    """
    Given one raw response string (containing multiple tasks in the numbered format),
    extract only the Task Title and Task Description (ignoring example questions).
    Returns a DataFrame with columns ["Task Title", "Task Description"].
    """
    rows = []
    pattern = r"\d+\.\s\*\*(.*?)\*\*:\s(.*?)(?=\n\s*-\s|\Z)"
    matches = re.findall(pattern, chunk_text, flags=re.DOTALL)

    for title, desc in matches:
        title = title.strip()
        desc = desc.strip()
        # Strip off any trailing “- Example…” lines from the description
        desc = re.sub(r"\n\s*-\s.*", "", desc, flags=re.DOTALL).strip()
        rows.append({
            "Task Title": title,
            "Task Description": desc
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

    df_list = []
    for chunk_idx, raw_text in response_dict.items():
        df_chunk = process_chunk(raw_text)
        # (Optional) Tag each row with its chunk index or any other metadata:
        df_chunk["chunk_index"] = chunk_idx
        df_list.append(df_chunk)

    if df_list:
        all_tasks_df = pd.concat(df_list, ignore_index=True)
        # Save to CSV (or JSON—your choice):
        all_tasks_df.to_csv("proposed_tasks.csv", index=False)
        print(f"✅ Saved {len(all_tasks_df)} total tasks to 'proposed_tasks.csv'")
    else:
        print("⚠️  No tasks extracted (response_dict was empty).")

if __name__ == "__main__":
    main()