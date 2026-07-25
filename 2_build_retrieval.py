import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

print("Chargement du modèle d'embedding (all-MiniLM-L6-v2)...")
# Ce modèle est très léger et tourne très bien sur CPU
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Chargement du corpus...")
corpus_df = pd.read_csv("data/corpus.csv")
evidences = corpus_df['evidence'].tolist()

print(f"Génération des embeddings pour {len(evidences)} preuves (sur CPU, cela prendra ~1-2 min)...")
# Encode the evidences
# normalize_embeddings=True permet d'utiliser le produit scalaire (Inner Product) comme similarité cosinus dans FAISS
embeddings = model.encode(evidences, show_progress_bar=True, normalize_embeddings=True)

print("Création de l'index FAISS...")
d = embeddings.shape[1] # Dimension des embeddings (384 pour MiniLM)
index = faiss.IndexFlatIP(d) # Inner Product -> equivalent à Cosine Similarity car normalisé
index.add(embeddings)

os.makedirs("models_saved", exist_ok=True)
faiss.write_index(index, "models_saved/faiss_index.bin")
print("Index FAISS sauvegardé dans 'models_saved/faiss_index.bin'.")

print("Terminé !")
