#!/usr/bin/env python3
"""
Build a Turso-compatible vector DB from triplets.parquet positives.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import turso
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(prog="build_vector_db.py")
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path("datasets/triplets.parquet"),
        help="Path to triplets.parquet file",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("db/vector.db"),
        help="Output SQLite vector DB path",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="./models/paraphrase-italian-mpnet-med-v2/final",
        help="HF model for embedding",
    )
    parser.add_argument(
        "--local-only",
        type=bool,
        default=True,
        help="Is model local",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding inference",
    )
    args = parser.parse_args()

    # Load positives column
    df = pl.read_parquet(args.parquet_path)
    positives = df["positive"].unique().to_list()

    # Compute embeddings
    print(f"Embedding {len(positives)} entries with {args.model_name}...")
    model = SentenceTransformer(
        args.model_name,
        local_files_only=args.local_only,
    )
    embeddings = model.encode(
        positives,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Ensure output directory exists
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to SQLite file
    print(f"Writing vector DB to {args.db_path}...")
    conn = turso.connect(args.db_path.as_posix())
    # Load vector extension if available (Turso WASM client will handle in-browser)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS positives (id INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT, embedding BLOB)"
    )
    conn.execute("DELETE FROM positives")
    for val, emb in zip(positives, embeddings):
        blob = emb.astype(np.float32).tobytes()
        conn.execute(
            "INSERT INTO positives(value, embedding) VALUES (?, vector32(?))",
            (val, blob),
        )
    conn.commit()
    conn.close()
    print("Vector DB build complete.")


if __name__ == "__main__":
    main()
