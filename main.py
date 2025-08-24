from sentence_transformers import SentenceTransformer
import numpy as np

sentences_dataset = [
    [
        "Acarbosio;USO ORALE;ACARBOSIO TECNIGEN;50 MG COMPRESSE 40 COMPRESSE IN BLISTER PVC/PCTFE/PVC/AL",
        "Acarbosio in compresse",
    ],
    [
        "Acarbosio;USO ORALE;ACARBOSIO TECNIGEN;50 MG COMPRESSE 40 COMPRESSE IN BLISTER PVC/PCTFE/PVC/AL",
        "Acarbosio",
    ],
    [
        "Acarbosio;USO ORALE;ACARBOSIO TECNIGEN;50 MG GOCCIE 40 COMPRESSE IN BLISTER PVC/PCTFE/PVC/AL",
        "Acarbosio",
    ],
    [
        "Acarbosio;USO ORALE;ACARBOSIO TECNIGEN;50 MG GOCCIE 40 COMPRESSE IN BLISTER PVC/PCTFE/PVC/AL",
        "Acarbosio in compresse",
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
