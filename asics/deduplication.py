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
    """
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
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    embeddings = []
    for text in df[column_name]:
        cleaned_text = clean_description(text)
        payload = {"model": "text-embedding-ada-002", "input": cleaned_text}
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json().get("data", [])
            embeddings.append(data[0]["embedding"] if data and "embedding" in data[0] else None)
        else:
            print(f"Error {response.status_code} embedding text: {cleaned_text[:50]}…")
            embeddings.append(None)

    df_copy = df.copy()
    df_copy["embedding"] = embeddings
    return df_copy


def remove_similar_rows(
    df: pd.DataFrame,
    title_column: str = "Task Title",
    embedding_column: str = "embedding",
    threshold: float = 0.88
) -> pd.DataFrame:
    """
    Removes rows whose cosine similarity exceeds the threshold.
    """
    valid_df = df[df[embedding_column].notna()].reset_index(drop=True)
    if valid_df.empty:
        return df

    matrix = np.vstack(valid_df[embedding_column].values)
    sim_matrix = cosine_similarity(matrix)
    drop_indices = set()
    for i in range(len(valid_df)):
        for j in range(i+1, len(valid_df)):
            if sim_matrix[i,j] > threshold:
                drop_indices.add(valid_df.index[j])
    return df.drop(index=list(drop_indices)).reset_index(drop=True)


def deduplicate(
    df: pd.DataFrame,
    threshold: float = 0.82,
    api_key: str = None
) -> pd.DataFrame:
    """
    Deduplicate category proposals by embedding titles and removing similar ones.
    :param df: DataFrame with 'Task Title' column.
    :param threshold: Cosine similarity threshold.
    :param api_key: OpenAI API key (optional, defaults to env var).
    :return: Deduplicated DataFrame without embeddings.
    """
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be set to run deduplication.")
    df_emb = embed_descriptions(df, "Task Title", key)
    df_dedup = remove_similar_rows(df_emb, threshold=threshold)
    return df_dedup.drop(columns=["embedding"])


def main():
    parser = argparse.ArgumentParser(description="Deduplicate tasks by cosine similarity.")
    parser.add_argument("--input-csv", type=str, required=True,
                        help="Path to CSV with columns 'Task Title'.")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output path; defaults to <input>_dedup.csv in ./output.")
    parser.add_argument("--threshold", type=float, default=0.82,
                        help="Cosine similarity threshold for dropping rows.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set.")

    df = pd.read_csv(args.input_csv)
    if "Task Title" not in df.columns:
        raise KeyError("Input CSV must contain 'Task Title'.")

    # Perform deduplication
    df_out = deduplicate(df, threshold=args.threshold, api_key=api_key)

    # Save to output directory
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.input_csv).stem
    out_path = Path(args.output_csv) if args.output_csv else out_dir / f"{base}_dedup.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved deduplicated categories ({len(df_out)} rows) to '{out_path}'")

if __name__ == "__main__":
    main()