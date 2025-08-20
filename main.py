from sentence_transformers import SentenceTransformer

sentences = [
    "Acarbosio;USO ORALE;ACARBOSIO TECNIGEN;50 MG COMPRESSE 40 COMPRESSE IN BLISTER PVC/PCTFE/PVC/AL",
    "Acarbosio in compresse",
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(sentences)
print(embeddings)
