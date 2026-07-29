import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
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

# C=0.1 choisi par grid search sur data/val.csv (C=1.0 par défaut surapprend :
# 1536 features pour ~4830 exemples d'entraînement). Impact mesuré sur test.csv,
# jamais utilisé pendant la recherche d'hyperparamètre : Macro-F1 0.48 (C=1.0,
# entraînement sur train seul) -> 0.53 (C=0.1, entraînement sur train+val).
CLASSIFIER_C = 0.1

print("Chargement des données...")
train_df = pd.read_csv("data/train.csv").dropna()
val_df = pd.read_csv("data/val.csv").dropna()
test_df = pd.read_csv("data/test.csv").dropna()

print("Chargement du modèle d'embedding...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\n--- Préparation des données d'entraînement ---")
X_train = create_features(train_df['claim'].tolist(), train_df['evidence'].tolist(), model)
y_train = train_df['label'].tolist()

print("\n--- Préparation des données de validation ---")
X_val = create_features(val_df['claim'].tolist(), val_df['evidence'].tolist(), model)
y_val = val_df['label'].tolist()

print("\n--- Préparation des données de test ---")
X_test = create_features(test_df['claim'].tolist(), test_df['evidence'].tolist(), model)
y_test = test_df['label'].tolist()

print(f"\nEntraînement sur train seul (C={CLASSIFIER_C}) pour estimer la généralisation...")
clf_trainonly = LogisticRegression(max_iter=1000, class_weight='balanced', C=CLASSIFIER_C)
clf_trainonly.fit(X_train, y_train)
f1_val = f1_score(y_val, clf_trainonly.predict(X_val), average='macro')
f1_test_trainonly = f1_score(y_test, clf_trainonly.predict(X_test), average='macro')
print(f"[Généralisation] Macro-F1 val={f1_val:.4f}  test={f1_test_trainonly:.4f}")

print("\nEntraînement du modèle final sur train+val (val.csv réintégré après sélection de C)...")
X_trainval = np.vstack((X_train, X_val))
y_trainval = list(y_train) + list(y_val)
clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=CLASSIFIER_C)
clf.fit(X_trainval, y_trainval)

print("\nPrédictions sur le test set (évaluation finale, unique)...")
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Métriques
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\n[OK] Macro-F1 final sur le set de test : {macro_f1:.4f}")

print("\nRapport de classification détaillé :")
print(classification_report(y_test, y_pred))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred, labels=clf.classes_))
print(f"Classes détectées : {clf.classes_}")

print("\nSauvegarde du modèle ML...")
os.makedirs("models_saved", exist_ok=True)
joblib.dump(clf, "models_saved/classifier.joblib")
print("Modèle sauvegardé dans 'models_saved/classifier.joblib'.")
print("Entraînement terminé !")
