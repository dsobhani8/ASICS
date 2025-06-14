#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportions_ztest

# Import your pipeline functions
from category_proposer import proposer        # returns DataFrame of proposed categories
from deduplication import deduplicate         # takes DataFrame, cosine threshold -> deduped DataFrame
from falsifier import run_falsifier           # takes df, categories -> full results DataFrame
from category_scorer import run_scorer       # takes df, categories -> scored DataFrame with 'Conclusion'


def compute_overall_score(df_scored: pd.DataFrame) -> float:
    """
    Compute overall fraction of 'Aligned' conclusions.
    """
    aligned_count = (df_scored['is_correct'] == True).sum()
    total = len(df_scored)
    return aligned_count / total if total > 0 else 0.0


def run_asics(diff_threshold: float, cosine_sim: float, percentage: float, input_dataset: str):
    # 1) Load original dataset
    data_dir = Path(os.path.expanduser("~/ASICS/data"))
    SPECIAL_FILES = {
        "hpd":      "hpd_llama_vs_claude.csv",
        "deepseek": "deepseek_coder-33b-instruct_bigcodebench_results.csv",
        # add more here as needed…
    }

    key = input_dataset.lower()
    filename = SPECIAL_FILES.get(key, f"{input_dataset}.csv")
    ds_file = data_dir / filename

    df = pd.read_csv(ds_file)

    # 2) Sample percentage of data
    if not (0.0 < percentage <= 100.0):
        raise ValueError("--percentage must be > 0 and ≤ 100")
    num_to_use = int(len(df) * (percentage / 100.0))
    df = df.iloc[:num_to_use].reset_index(drop=True)
    print(f"Using {len(df)} rows ({percentage}%) of dataset for pipeline.")

    # 3) Split into X (30%) and Y (70%)
    x_df, y_df = train_test_split(df, test_size=0.7, random_state=42)
    print(f"Split: {len(x_df)} rows for refinement, {len(y_df)} rows for validation.")

    # 4) Propose categories on X
    proposed = proposer(x_df)

    # 5) Deduplicate proposals
    deduped = deduplicate(proposed, threshold=cosine_sim)
    print(f"Deduplicated to {len(deduped)} unique categories.")

    # 6) Falsifier on X
    falsifier_results = run_falsifier(x_df, deduped)

    # 7) Scorer on X against falsified
    scored_x = run_scorer(x_df, falsifier_results)
    overall_x = compute_overall_score(scored_x)
    print(f"Overall aligned rate on X: {overall_x:.3f}")

    # 8) Compute category-wise scores and differences
    stats = []
    for _, row in deduped.iterrows():
        title = row['Task Title']
        desc = row['Task Description']
        subset = scored_x[(scored_x['Task Title']==title) & (scored_x['Task Description']==desc)]
        n = len(subset)
        k = (subset['is_correct']==True).sum()
        p = k / n if n>0 else 0.0
        stats.append({'Task Title': title, 'Task Description': desc, 'n': n, 'k': k, 'p': p})
    cat_df = pd.DataFrame(stats)
    cat_df['diff'] = cat_df['p'] - overall_x

    # 9) Refine: select categories where diff > threshold
    refined = cat_df[cat_df['diff'] > diff_threshold].copy()
    out_dir = Path('output'); out_dir.mkdir(exist_ok=True)
    refined_file = out_dir / f"{input_dataset}_refined_categories.csv"
    refined[['Task Title','Task Description']].to_csv(refined_file, index=False)
    print(f"Refined {len(refined)} categories (diff > {diff_threshold}) -> {refined_file}")

    # 10) Validate on Y
    # scored_y = run_scorer(y_df, refined[['Task Title','Task Description']])
    scored_y = run_scorer(y_df, falsifier_results)
    print(scored_y)
    overall_y = compute_overall_score(scored_y)
    print(f"Overall aligned rate on Y: {overall_y:.3f}")

    # 11) Test significance per refined category on Y
    results = []
    # for _, r in refined.iterrows():
    for _, r in falsifier_results.iterrows():
        title, desc = r['Task Title'], r['Task Description']
        subset = scored_y[(scored_y['Task Title']==title) & (scored_y['Task Description']==desc)]
        n_y = len(subset)
        k_y = (subset['is_correct']==True).sum()
        cat_score=k_y/n_y
        if n_y > 0:
            stat, pval = proportions_ztest(k_y, n_y, value=overall_y)
        else:
            stat, pval = (None, None)
        results.append({
            'Task Title': title,
            'Task Description': desc,
            'n_y': n_y,
            'k_y': k_y,
            'z': stat,
            'pvalue': pval,
            'category_score': cat_score
        })
    sig_df = pd.DataFrame(results)
    
    sig_file = Path(args.output_csv) if args.output_csv else out_dir / f"{args.input_dataset}_significance_results.csv"
    sig_df.to_csv(sig_file, index=False)
    print(f"Saved significance tests for {len(sig_df)} categories -> {sig_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full ASICS pipeline and refine categories.")
    parser.add_argument("--difference-threshold", type=float, default=0.05,
                        help="Minimum difference (category rate - overall rate) to keep a category.")
    parser.add_argument("--cosine-sim", type=float, default=0.82,
                        help="Cosine similarity threshold for deduplication.")
    parser.add_argument("--percentage", type=float, default=100.0,
                        help="Percentage of the input dataset to use (0 < percentage ≤ 100). Default 100.")
    parser.add_argument("--input_dataset", type=str, required=True,
                        help="Dataset name (without .csv). 'hpd' uses hpd_llama_vs_claude.csv.")
    parser.add_argument("--output_csv", type=str, required=False,
                        help="output dataset")
    args = parser.parse_args()
    run_asics(args.difference_threshold, args.cosine_sim, args.percentage, args.input_dataset)
