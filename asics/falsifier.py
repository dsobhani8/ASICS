#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import openai
import time
import re
from pathlib import Path
from query import chat_completion


def load_dataset(dataset_name: str) -> pd.DataFrame:
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
    if not categories_csv.is_file():
        raise FileNotFoundError(f"Could not find categories file at {categories_csv}")
    df = pd.read_csv(categories_csv)
    required = {"Task Title", "Task Description"}
    if not required.issubset(df.columns):
        raise KeyError(f"Categories CSV must contain columns: {required}")
    return df.drop_duplicates(subset=['Task Title','Task Description']).reset_index(drop=True)


def load_prompt_template() -> str:
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


def run_falsifier(
    df_questions: pd.DataFrame,
    df_categories: pd.DataFrame,
    percentage: float = 100.0,
    pause_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Run the falsifier and return only the pruned categories
    (those with uncertain_rate <= 10%).

    Returns a DataFrame with columns ['Task Title','Task Description'].
    """
    # slice questions
    if not (0.0 < percentage <= 100.0):
        raise ValueError("percentage must be > 0 and <= 100")
    total_q = len(df_questions)
    num_to_process = int(total_q * (percentage / 100.0))
    df_q = df_questions.iloc[:num_to_process].reset_index(drop=True)

    # load prompt
    prompt_template = load_prompt_template()

    # collect responses
    records = []
    for question in df_q['system_input'].astype(str):
        for _, row in df_categories.iterrows():
            title = row['Task Title']
            desc  = row['Task Description']
            prompt = (
                prompt_template
                .replace('{Category Title}', title)
                .replace('{Category Descrip}', desc)
                .replace('{question}', question)
            )
            try:
                # response = openai.chat.completions.create(
                #     model=os.getenv('MODEL_NAME'),
                #     messages=[
                #         {"role":"system","content":"You are an expert at identifying categories in questions."},
                #         {"role":"user","content":prompt}
                #     ]
                # )
                response = chat_completion(prompt)

                text = response.choices[0].message.content
            except Exception:
                text = ""
            m = re.search(r'(?i)Conclusion:\s*(aligned|unaligned|uncertain)\b', text)
            conclusion = m.group(1).capitalize() if m else None
            records.append({'Task Title': title, 'Task Description': desc, 'Conclusion': conclusion})
            time.sleep(pause_seconds)

    # build DataFrame and compute uncertain rates
    df_out = pd.DataFrame(records)
    stats = df_out.groupby(['Task Title','Task Description'])['Conclusion']
    total = stats.size()
    uncertain = stats.apply(lambda s: (s=='Uncertain').sum())
    rate = (uncertain / total).rename('uncertain_rate')
    combined = pd.concat([total.rename('total'), uncertain.rename('uncertain'), rate], axis=1)

    # select pruned categories
    pruned = combined[combined['uncertain_rate'] <= 0.10].reset_index()[['Task Title','Task Description']]
    return pruned


def main():
    parser = argparse.ArgumentParser(description="Run falsifier and output pruned categories.")
    parser.add_argument("--dataset",      type=str, required=True)
    parser.add_argument("--categories-csv", type=str, required=True)
    parser.add_argument("--percentage",   type=float, default=100.0)
    parser.add_argument("--output-csv",   type=str, default=None)
    parser.add_argument("--pause-seconds",type=float, default=1.0)
    args = parser.parse_args()

    openai.api_key = os.getenv('OPENAI_API_KEY','')
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    df_q = load_dataset(args.dataset)
    df_c = load_categories(Path(args.categories_csv))

    pruned = run_falsifier(df_q, df_c, percentage=args.percentage, pause_seconds=args.pause_seconds)

    out_dir = Path('output'); out_dir.mkdir(exist_ok=True)
    out_path = Path(args.output_csv) if args.output_csv else out_dir / f"{args.dataset}_falsified_categories.csv"
    pruned.to_csv(out_path, index=False)
    print(f"Saved pruned categories ({len(pruned)} rows) to '{out_path}'")

if __name__ == "__main__":
    main()
