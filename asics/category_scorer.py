#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import openai
import time
from pathlib import Path
import re
from query import chat_completion

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load the original evaluation dataset CSV from ~/ASICS/data/.
    If dataset_name == "hpd", loads "hpd_llama_vs_claude.csv"; otherwise "{dataset_name}.csv".
    Expects a "system_input" column containing the questions.
    """
    data_dir = Path(os.path.expanduser("~/ASICS/data"))
    filename = "hpd_llama_vs_claude.csv" if dataset_name.lower() == "hpd" else f"{dataset_name}.csv"
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
    return df.drop_duplicates(subset=['Task Title','Task Description']).reset_index(drop=True)


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


def run_scorer(
    df_questions: pd.DataFrame,
    df_categories: pd.DataFrame,
    percentage: float = 100.0,
    pause_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Programmatic interface: run the category scorer logic and return a DataFrame with columns:
      ['question_index','question','Task Title','Task Description','full_scorer_response','short_scorer_response','is_correct']
    """
    # 1) Slice questions by percentage
    if not (0.0 < percentage <= 100.0):
        raise ValueError("percentage must be > 0 and <= 100")
    total_q = len(df_questions)
    num_to_process = int(total_q * (percentage / 100.0))
    df_q = df_questions.iloc[:num_to_process].reset_index(drop=True)
    print(df_q.columns)

    # 2) Load prompt template
    prompt_template = load_prompt_template()

    # 3) Iterate and call API
    records = []
    for q_idx, question in enumerate(df_q['system_input'].astype(str), start=1):
        answer = df_q.at[q_idx-1, 'is_correct']

        for _, row in df_categories.iterrows():
            title = row['Task Title'].strip()
            desc  = row['Task Description'].strip()
            prompt = (
                prompt_template
                .replace('{task_title}', title)
                .replace('{task_description}', desc)
                .replace('{question}', question)
            )
            try:
                # respoonse = openai.chat.completions.create(
                #     model=os.getenv('MODEL_NAME'),
                #     messages=[
                #         {"role": "system",  "content": "You are an expert at identifying categories in questions."},
                #         {"role": "user",    "content": prompt}
                #     ]
                # )
                response = chat_completion(prompt)

                reply = response.choices[0].message.content
            except Exception:
                reply = ""
            # parse short response
            m = re.search(r'(?i)Conclusion:\s*(yes|no)\b', reply)
            short = m.group(1).capitalize() if m else None

            records.append({
                'question_index': q_idx - 1,
                'question': question,
                'Task Title': title,
                'Task Description': desc,
                'full_scorer_response': reply,
                'Conclusion': short,
                'is_correct': answer

            })
            time.sleep(pause_seconds)
    return pd.DataFrame.from_records(records)


def main():
    parser = argparse.ArgumentParser(
        description="Run the category scorer on each (question, category) pair."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Name of the dataset (without .csv); e.g. 'hpd'."
    )
    parser.add_argument(
        "--categories-csv", type=str, required=True,
        help="Path to the deduplicated categories CSV."
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
        help="Seconds to sleep between API calls."
    )
    args = parser.parse_args()

    # setup
    openai.api_key = os.getenv('OPENAI_API_KEY', '')
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # load inputs
    df_q = load_dataset(args.dataset)
    df_c = load_categories(Path(args.categories_csv))

    # run
    df_out = run_scorer(
        df_q,
        df_c,
        percentage=args.percentage,
        pause_seconds=args.pause_seconds
    )

    # save
    out_dir = Path('output')
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = args.output_csv or f"{args.dataset}_scorer.csv"
    out_path = out_dir / fname
    df_out.to_csv(out_path, index=False)
    print(f"Saved scorer results ({len(df_out)} rows) to '{out_path}'")

if __name__ == '__main__':
    main()