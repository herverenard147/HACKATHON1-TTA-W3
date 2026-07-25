import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("models_saved/faiss_index.bin")
df = pd.read_csv("data/corpus.csv")

claim = "la Côte d'ivoire est un pays chaud!"
c_emb = model.encode([claim], normalize_embeddings=True)

k = 3
distances, indices = index.search(c_emb, k)

for i in range(k):
    print(f"Match {i+1}: Score = {distances[0][i]:.4f}")
    print(f"Evidence: {df.iloc[indices[0][i]]['evidence']}")
    print("-" * 50)
