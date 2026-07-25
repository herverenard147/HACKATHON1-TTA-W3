import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, brier_score_loss, confusion_matrix
import joblib
import os

def create_features(claims, evidences, model):
    print("  Encodage des claims...")
    c_emb = model.encode(claims, show_progress_bar=True, normalize_embeddings=True)
    print("  Encodage des evidences...")
    e_emb = model.encode(evidences, show_progress_bar=True, normalize_embeddings=True)
    
    # Feature engineering classique pour le NLI (Natural Language Inference)
    abs_diff = np.abs(c_emb - e_emb)
    elementwise_mult = c_emb * e_emb
    
    # Concaténation des vecteurs
    features = np.hstack((c_emb, e_emb, abs_diff, elementwise_mult))
    return features

print("Chargement des données...")
train_df = pd.read_csv("data/train.csv").dropna()
val_df = pd.read_csv("data/val.csv").dropna()
test_df = pd.read_csv("data/test.csv").dropna()

print("Chargement du modèle d'embedding...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\n--- Préparation des données d'entraînement ---")
X_train = create_features(train_df['claim'].tolist(), train_df['evidence'].tolist(), model)
y_train = train_df['label'].tolist()

print("\n--- Préparation des données de test ---")
X_test = create_features(test_df['claim'].tolist(), test_df['evidence'].tolist(), model)
y_test = test_df['label'].tolist()

print("\nEntraînement de la Régression Logistique (très rapide sur CPU)...")
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

print("\nPrédictions sur le test set...")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Métriques
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n[OK] Macro-F1 sur le set de test : {macro_f1:.4f}")

print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred))

# Évaluation de la Calibration
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred, labels=clf.classes_))
print(f"Classes détectées : {clf.classes_}")

print("\nSauvegarde du modèle ML...")
os.makedirs("models_saved", exist_ok=True)
joblib.dump(clf, "models_saved/classifier.joblib")
print("Modèle sauvegardé dans 'models_saved/classifier.joblib'.")
print("Entraînement terminé !")
