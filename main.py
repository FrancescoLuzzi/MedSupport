import numpy as np
from sentence_transformers import SentenceTransformer

sentences_dataset = [
    [
        "acarbosio;uso orale;acarbosio tecnigen;50 mg compresse 40 compresse in blister pvc/pctfe/pvc/al",
        "acarbosio in compresse",
    ],
    [
        "acarbosio;uso orale;acarbosio tecnigen;50 mg compresse 40 compresse in blister pvc/pctfe/pvc/al",
        "acarbosio",
    ],
    [
        "acarbosio;uso orale;acarbosio tecnigen;50 mg goccie 40 compresse in blister pvc/pctfe/pvc/al",
        "acarbosio",
    ],
    [
        "acarbosio;uso orale;acarbosio tecnigen;50 mg goccie 40 compresse in blister pvc/pctfe/pvc/al",
        "acarbosio in compresse",
    ],
]


def cos_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


paraphrase_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
# paraphrase_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

all_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

for sentences in sentences_dataset:
    paraphrase_embeddings = paraphrase_model.encode(sentences)
    all_embeddings = all_model.encode(sentences)

    print(
        f"""sentence= {sentences[0]}
query = {sentences[1]}
all_similarity = {cos_similarity(*all_embeddings)}
paraphrase_similarity={cos_similarity(*paraphrase_embeddings)}"""
    )
