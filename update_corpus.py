import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

print("Chargement des corpus...")
corpus_original = pd.read_csv("data/corpus.csv")
corpus_add = pd.read_csv("data/corpus_additionnel.csv")

print(f"Fusion : {len(corpus_original)} originaux + {len(corpus_add)} additionnels")
corpus_merged = pd.concat([corpus_original, corpus_add]).drop_duplicates(subset=['evidence']).reset_index(drop=True)

# Sauvegarde du nouveau corpus global
corpus_merged.to_csv("data/corpus.csv", index=False)
print(f"Corpus mis à jour. Taille totale : {len(corpus_merged)}")

print("Chargement du modèle d'embedding (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

evidences = corpus_merged['evidence'].tolist()

print(f"Régénération des embeddings pour {len(evidences)} preuves...")
embeddings = model.encode(evidences, show_progress_bar=True, normalize_embeddings=True)

print("Création de l'index FAISS...")
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

os.makedirs("models_saved", exist_ok=True)
faiss.write_index(index, "models_saved/faiss_index.bin")
print("Index FAISS mis à jour et sauvegardé dans 'models_saved/faiss_index.bin'.")
