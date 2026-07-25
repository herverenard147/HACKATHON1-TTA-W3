from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import joblib

app = FastAPI(
    title="ClimaCheck API",
    description="API de vérification des faits climatiques (Zéro-GPU)",
    version="1.0"
)

# Modèle de requête
class ClaimRequest(BaseModel):
    claim: str

# Variables globales pour les modèles
embedding_model = None
index = None
classifier = None
corpus_df = None

@app.on_event("startup")
def load_models():
    global embedding_model, index, classifier, corpus_df
    print("Chargement des modèles en mémoire...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        index = faiss.read_index("models_saved/faiss_index.bin")
        classifier = joblib.load("models_saved/classifier.joblib")
        corpus_df = pd.read_csv("data/corpus.csv")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")

@app.post("/verify")
def verify_claim(request: ClaimRequest):
    if not request.claim:
        raise HTTPException(status_code=400, detail="L'affirmation ne peut pas être vide")
        
    try:
        c_emb = embedding_model.encode([request.claim], normalize_embeddings=True)
        
        # FAISS search
        k = 1
        distances, indices = index.search(c_emb, k)
        top_evidence = corpus_df.iloc[indices[0][0]]['evidence']
        similarity_score = float(distances[0][0])
        
        if similarity_score < 0.45:
            verdict = "NOT_ENOUGH_INFO"
            confidence = 0.0
        else:
            # Feature eng
            e_emb = embedding_model.encode([top_evidence], normalize_embeddings=True)
            abs_diff = np.abs(c_emb - e_emb)
            elementwise_mult = c_emb * e_emb
            features = np.hstack((c_emb, e_emb, abs_diff, elementwise_mult))
            
            # Prédiction
            verdict = classifier.predict(features)[0]
            probas = classifier.predict_proba(features)[0]
            confidence = float(np.max(probas) * 100)
        
        return {
            "claim": request.claim,
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "top_evidence": top_evidence,
            "similarity_score": float(distances[0][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
