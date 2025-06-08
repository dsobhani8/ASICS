#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import openai
import time
from pathlib import Path

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load the original evaluation dataset CSV from ~/ASICS/data/.
    If dataset_name == "hpd", loads "hpd_llama_vs_claude.csv"; otherwise "{dataset_name}.csv".
    Expects a "system_input" column containing the questions.
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
    Load the falsified (deduplicated) categories CSV.
    Expects columns "Task Title", "Task Description".
    """
    if not categories_csv.is_file():
        raise FileNotFoundError(f"Could not find categories file at {categories_csv}")
    df = pd.read_csv(categories_csv)
    required = {"Task Title", "Task Description"}
    if not required.issubset(df.columns):
        raise KeyError(f"Categories CSV must contain columns: {required}")
    
    if {'Task Title', 'Task Description'}.issubset(df.columns):
        return df.drop_duplicates(subset=['Task Title', 'Task Description']).reset_index(drop=True)


def load_prompt_template() -> str:
    """
    Starting from this script's folder, walk upward until finding "prompts/category_scorer.txt".
    Returns its contents as a string.
    """
    current = Path(__file__).resolve().parent
    while True:
        candidate = current / "prompts" / "category_scorer.txt"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        f"Could not locate 'prompts/category_scorer.txt' above {Path(__file__).resolve()}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the category scorer on each (question, category) pair."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Name of the dataset (without .csv); e.g. 'hpd' to load hpd_llama_vs_claude.csv."
    )
    parser.add_argument(
        "--categories-csv", type=str, required=True,
        help="Path to the deduplicated categories CSV (e.g. hpd_proposed_categories_dedup.csv)."
    )
    parser.add_argument(
        "--percentage", type=float, default=100.0,
        help="Percentage of questions to process (0 < percentage ≤ 100). Default: 100."
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Where to write the scorer output. Defaults to <dataset>_scorer.csv"
    )
    parser.add_argument(
        "--pause-seconds", type=float, default=1.0,
        help="Seconds to sleep between API calls to avoid rate limits."
    )
    args = parser.parse_args()

    # 1) Load API key and model
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model_name = os.getenv("MODEL_NAME", "")
    if not model_name:
        raise RuntimeError("MODEL_NAME is not set.")

    # 2) Load questions
    df_questions = load_dataset(args.dataset)
    if not (0.0 < args.percentage <= 100.0):
        raise ValueError("--percentage must be > 0 and ≤ 100")
    total_q = len(df_questions)
    num_to_process = int(total_q * (args.percentage / 100.0))
    df_questions = df_questions.iloc[:num_to_process].reset_index(drop=True)
    print(f"Processing {num_to_process} of {total_q} questions ({args.percentage}%).")

    # 3) Load categories
    df_categories = load_categories(Path(args.categories_csv))
    total_c = len(df_categories)
    print(f"Loaded {total_c} categories → evaluating {num_to_process * total_c} pairs.")

    # 4) Load prompt template
    prompt_template = load_prompt_template()

    # 5) Iterate and call API
    records = []
    for q_idx, question in enumerate(df_questions['system_input'].astype(str), start=1):
        for c_idx, row in df_categories.iterrows():
            title = row['Task Title'].strip()
            desc = row['Task Description'].strip()
            prompt = (
                prompt_template
                .replace('{task_title}', title)
                .replace('{task_description}', desc)
                .replace('{question}', question)
            )
            try:
                response = openai.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying categories in questions."},
                        {"role": "user", "content": prompt}
                    ]
                )
                reply = response.choices[0].message.content
            except Exception as e:
                print(f"[Q{q_idx}/C{c_idx}] API error: {e}")
                reply = ""

            records.append({
                'question_index': q_idx - 1,
                'question': question,
                # 'category_index': c_idx,
                'Task Title': title,
                'Task Description': desc,
                'scorer_response': reply
            })
            time.sleep(args.pause_seconds)

    # 6) Save results

    df_out = pd.DataFrame.from_records(records)
    df_out['Scorer_Response'] = (
    df_out['scorer_response']
      .str.extract(r'(?i)Conclusion:\s*(yes|no)\b', expand=False)
      .str.capitalize()
    )
    output_dir = Path("output")  
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = args.output_csv if args.output_csv else f"{args.dataset}_scorer.csv"

    out_path = output_dir / filename
    df_out.to_csv(out_path, index=False)
    print(f"Saved scorer results ({len(df_out)} rows) to '{out_path}'")

if __name__ == '__main__':
    main()
