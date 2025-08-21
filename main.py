from sentence_transformers import SentenceTransformer
import numpy as np
import polars as pd
from pathlib import Path

training_data_folder = Path("./data")
training_dataset_file = training_data_folder / "lista_farmaci_equivalenti.csv"

dataset = pd.read_csv(training_dataset_file, separator=";")
print(dataset["Principio attivo"].unique())

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
