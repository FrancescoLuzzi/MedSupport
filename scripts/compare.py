#!/usr/bin/env python3

import numpy as np
from sentence_transformers import SentenceTransformer


def cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    sentences_dataset = [
        [
            "acarbosio in compresse",
            "acarbosio;uso orale;acarbosio tecnigen;50 mg compresse 40 compresse in blister pvc/pctfe/pvc/al",
        ],
        [
            "acarbosio",
            "acarbosio;uso orale;acarbosio tecnigen;50 mg compresse 40 compresse in blister pvc/pctfe/pvc/al",
        ],
        [
            "acarbosio",
            "acarbosio;uso orale;acarbosio tecnigen;50 mg goccie 40 compresse in blister pvc/pctfe/pvc/al",
        ],
        [
            "carcsiso in compresse",
            "acarbosio;uso orale;acarbosio tecnigen;50 mg goccie 40 compresse in blister pvc/pctfe/pvc/al",
        ],
    ]

    paraphrase_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    tuned_model = SentenceTransformer(
        "./models/paraphrase-italian-mpnet-med-v2/final",
        local_files_only=True,
    )

    for sentences in sentences_dataset:
        paraphrase_embeddings = paraphrase_model.encode(sentences)
        all_embeddings = tuned_model.encode(sentences)

        print(
            f"""query = {sentences[1]}
    sentence= {sentences[0]}
    tuned_similarity = {cos_similarity(*all_embeddings)}
    old_similarity={cos_similarity(*paraphrase_embeddings)}"""
        )


if __name__ == "__main__":
    main()
