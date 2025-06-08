#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import openai
import time
from pathlib import Path

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Loads the original dataset CSV from ~/ASICS/data/.
    If dataset_name == "hpd", loads "hpd_llama_vs_claude.csv"; otherwise "{dataset_name}.csv".
    Expects a "system_input" column.
    """
    data_dir = Path(os.path.expanduser("~/ASICS/data"))
    if dataset_name.lower() == "hpd":
        filename = "hpd_llama_vs_claude.csv"
    else:
        filename = f"{dataset_name}.csv"

    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Could not find dataset at {path}")
    df = pd.read_csv(path)
    if "system_input" not in df.columns:
        raise KeyError(f"'system_input' column not found in {path}")
    return df

def load_categories(categories_csv: Path) -> pd.DataFrame:
    """
    Loads the deduplicated categories CSV.
    Expects at least columns "Task Title" and "Task Description".
    """
    if not categories_csv.is_file():
        raise FileNotFoundError(f"Could not find categories file at {categories_csv}")
    df = pd.read_csv(categories_csv)
    required = {"Task Title", "Task Description"}
    if not required.issubset(df.columns):
        raise KeyError(f"Categories CSV must contain columns: {required}")
    return df

def load_prompt_template() -> str:
    """
    Starting from this script’s folder, walk upward until finding "prompts/falsifier.txt".
    Returns its contents as a string.
    """
    current = Path(__file__).resolve().parent
    while True:
        candidate = current / "prompts" / "falsifier.txt"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError(
        f"Could not locate 'prompts/falsifier.txt' above {Path(__file__).resolve()}"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Run the falsifier on each (question, category) pair."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (without .csv); e.g. 'hpd' to load hpd_llama_vs_claude.csv."
    )
    parser.add_argument(
        "--categories-csv",
        type=str,
        required=True,
        help="Path to the deduplicated categories CSV (e.g. hpd_proposed_categories_dedup.csv)."
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Percentage of questions to process (0 < percentage ≤ 100). Default: 100."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Where to write the falsifier output. Defaults to <dataset>_falsifier.csv"
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls (to avoid rate limits)."
    )
    args = parser.parse_args()

    # 1) Load env vars for OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model_name = os.getenv("MODEL_NAME", "")
    if not model_name:
        raise RuntimeError("MODEL_NAME is not set.")

    # 2) Load dataset and apply percentage slicing
    try:
        df_questions = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if not (0.0 < args.percentage <= 100.0):
        raise ValueError("--percentage must be > 0 and ≤ 100")
    total_q = len(df_questions)
    num_to_process = int(total_q * (args.percentage / 100.0))
    df_questions = df_questions.iloc[:num_to_process].reset_index(drop=True)
    print(f"Processing {num_to_process} of {total_q} questions ({args.percentage}%).")

    try:
        df_categories = load_categories(Path(args.categories_csv))
    except Exception as e:
        print(f"Error loading categories: {e}")
        return

    # 3) Load falsifier prompt template
    try:
        prompt_template = load_prompt_template()
    except Exception as e:
        print(f"Error loading falsifier prompt: {e}")
        return

    # 4) Prepare output storage
    records = []
    total_c = len(df_categories)
    print(f"{total_c} categories loaded → {len(df_questions) * total_c} pairs to evaluate.")

    # 5) Iterate over every (question, category) pair
    for q_idx, question in enumerate(df_questions["system_input"].astype(str), start=1):
        for c_idx, row in df_categories.iterrows():
            task_mode = str(row["Task Title"]).strip()
            task_desc = str(row["Task Description"]).strip()

            # print('TASK', task_mode)
            # print('DESC', task_desc)
            # Substitute into the prompt
            prompt = (
                prompt_template
                .replace("{Category Title}", task_mode)
                .replace("{Category Descrip}", task_desc)
                .replace("{question}", question)
            )
            print(prompt)
            # Call the model
            try:
                response = openai.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=[
                    {"role": "system", "content": "You are an expert at identifying categories in questions."},
                    {"role": "user", "content": prompt}
                    ]
                )
                reply = response.choices[0].message.content
            except Exception as api_err:
                print(f"[Q{q_idx}/C{c_idx}] API error: {api_err}")
                reply = ""

            # Store record
            records.append({
                # "question_index": q_idx - 1,
                "question": question,
                # "category_index": c_idx,
                "Task Title": task_mode,
                "Task Description": task_desc,
                "falsifier_response": reply
            })

            # Pause briefly to avoid hitting rate limits
            time.sleep(args.pause_seconds)

    # 6) Build DataFrame and save
    df_out = pd.DataFrame.from_records(records)
    df_out['Conclusion'] = (
    df_out['falsifier_response']
      .str.extract(r'(?i)Conclusion:\s*(aligned|unaligned|uncertain)\b', expand=False)
      .str.capitalize()
    )
    output_dir = Path("output")  
    output_dir.mkdir(parents=True, exist_ok=True)

    # if args.output_csv:
    #     out_path = Path(args.output_csv)
    # else:
    #     out_path = Path(f"{args.dataset}_falsifier.csv")
    filename = args.output_csv if args.output_csv else f"{args.dataset}_falsifier.csv"

    out_path = output_dir / filename
    df_out.to_csv(out_path, index=False)
    print(f"Saved falsifier results ({len(df_out)} rows) to '{out_path}'.")

if __name__ == "__main__":
    main()
