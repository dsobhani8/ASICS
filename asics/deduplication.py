#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

def clean_description(description: str) -> str:
    """
    Removes leading numbering and any special formatting from the description.

    :param description: A string representing the description.
    :return: A cleaned string of the description.
    """
    # Remove numbering like '1. ' or '23. '
    cleaned = re.sub(r'^\d+\.\s*', '', description).strip()
    return cleaned

def embed_descriptions(
    df: pd.DataFrame,
    column_name: str,
    api_key: str,
    api_url: str = "https://api.openai.com/v1/embeddings"
) -> pd.DataFrame:
    """
    Embeds each text in df[column_name] using the OpenAI embeddings endpoint.

    :param df: DataFrame containing the column to embed.
    :param column_name: Name of the column with text to embed.
    :param api_key: OpenAI API key.
    :param api_url: URL for the OpenAI embeddings endpoint.
    :return: The DataFrame with an added 'embedding' column.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    embeddings = []
    for text in df[column_name]:
        cleaned_text = clean_description(text)
        payload = {
            "model": "text-embedding-ada-002",
            "input": cleaned_text
        }

        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data and "embedding" in data[0]:
                embeddings.append(data[0]["embedding"])
            else:
                embeddings.append(None)
                print(f"Warning: no 'embedding' field in response for text: {cleaned_text[:50]}…")
        else:
            embeddings.append(None)
            print(f"Error {response.status_code} embedding text: {cleaned_text[:50]}…\n{response.text}")

    df = df.copy()
    df["embedding"] = embeddings
    return df

def remove_similar_rows(
    df: pd.DataFrame,
    title_column: str = "Task Title",
    embedding_column: str = "embedding",
    threshold: float = 0.88
) -> pd.DataFrame:
    """
    Removes rows whose cosine similarity (based on embeddings) exceeds the threshold.

    :param df: DataFrame containing an 'embedding' column (list of floats) and a title column.
    :param title_column: Name of the column holding the text (for reporting).
    :param embedding_column: Name of the column holding embeddings.
    :param threshold: Cosine similarity threshold above which to drop a row.
    :return: Filtered DataFrame with similar rows removed.
    """
    # Filter out any rows where embedding is None
    valid_df = df[df[embedding_column].notna()].reset_index(drop=True)
    if valid_df.empty:
        print("No valid embeddings to compute similarities. Returning original DataFrame.")
        return df

    # Build an (n × d) matrix from the 'embedding' lists
    embeddings_matrix = np.vstack(valid_df[embedding_column].values)
    sim_matrix = cosine_similarity(embeddings_matrix)

    rows_to_drop = set()
    high_sim_pairs = []

    n = len(valid_df)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > threshold:
                # Mark the later index for removal
                rows_to_drop.add(valid_df.index[j])
                # Record the pair for logging
                high_sim_pairs.append((
                    valid_df.iloc[i][title_column],
                    valid_df.iloc[j][title_column],
                    sim_matrix[i, j]
                ))

    if rows_to_drop:
        print("\nRows to be dropped (index → title):")
        for idx in sorted(rows_to_drop):
            print(f"  {idx} → {df.loc[idx, title_column]}")
        print("\nHigh-similarity pairs (title1, title2, score):")
        for t1, t2, score in high_sim_pairs:
            print(f"  ({score:.4f})\n    • {t1}\n    • {t2}\n")
    else:
        print("No pairs exceeded the similarity threshold. No rows dropped.")

    # Drop the marked rows from the original DataFrame
    df_filtered = df.drop(index=list(rows_to_drop)).reset_index(drop=True)
    return df_filtered

def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate tasks by cosine similarity on embeddings."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the CSV file produced by category_proposer (e.g., hpd_proposed_categories.csv)."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save the deduplicated CSV. Defaults to <dataset>_dedup.csv"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold above which to drop a row (default: 0.88)."
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    # Load the input CSV
    df = pd.read_csv(args.input_csv)
    if "Task Title" not in df.columns:
        raise KeyError("Input CSV must contain a 'Task Title' column.")

    print(f"Loaded {len(df)} rows from '{args.input_csv}'.")

    # Embed the Task Title column
    df_embedded = embed_descriptions(df, "Task Title", api_key)

    # Remove duplicates
    df_dedup = remove_similar_rows(df_embedded, threshold=args.threshold)
    df_dedup = df_dedup.drop(columns=["embedding"])

    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # results_path = output_dir / f"{args.dataset}_dedup.csv"
    # df_dedup.to_csv(results_path, index=False)
    # print(f"Saved full falsifier results ({len(df_dedup)} rows) to '{results_path}'")
    # Decide output filename
    if args.output_csv:
        out_path = args.output_csv
    else:
        base = Path(args.input_csv).stem
        out_path = output_dir / f"{base}_dedup.csv"

    df_dedup.to_csv(out_path, index=False)
    print(f"\nSaved deduplicated results ({len(df_dedup)} rows) to '{out_path}'.")

if __name__ == "__main__":
    main()
