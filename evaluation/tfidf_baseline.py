# Baseline de comparaison pour le classificateur de production (voir
# DOCUMENTATION_TECHNIQUE.md, section "Métriques réelles mesurées") :
# remplace l'encodeur de phrases (all-MiniLM-L6-v2) par une simple
# vectorisation TF-IDF, en gardant la même architecture de feature
# engineering (concat + |diff| + produit) et le même protocole d'évaluation
# (tuning de C sur val.csv, entraînement final sur train+val, une seule
# évaluation sur test.csv). Sert à mesurer l'apport réel des embeddings
# sémantiques par rapport à une approche lexicale classique.
#
# À lancer depuis la racine du dépôt (comme les autres scripts) :
#   source venv/bin/activate
#   python3 evaluation/tfidf_baseline.py
#
# Dernier résultat mesuré (voir DOCUMENTATION_TECHNIQUE.md) : Macro-F1 test = 0.4854

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

train_df = pd.read_csv("data/train.csv").dropna()
val_df = pd.read_csv("data/val.csv").dropna()
test_df = pd.read_csv("data/test.csv").dropna()

# Meme architecture de feature engineering (concat, |diff|, produit) mais avec
# TF-IDF (max_features=2000) a la place des embeddings de phrases, pour une
# comparaison a armes egales sur le meme pipeline de classification.
vectorizer = TfidfVectorizer(max_features=2000)
all_text = pd.concat([train_df['claim'], train_df['evidence']])
vectorizer.fit(all_text)

def make_features(df):
    c = vectorizer.transform(df['claim']).toarray()
    e = vectorizer.transform(df['evidence']).toarray()
    abs_diff = np.abs(c - e)
    mult = c * e
    return np.hstack([c, e, abs_diff, mult])

X_train = make_features(train_df)
X_val = make_features(val_df)
X_test = make_features(test_df)
y_train, y_val, y_test = train_df['label'].tolist(), val_df['label'].tolist(), test_df['label'].tolist()

best_C, best_f1v = None, -1
for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=C)
    clf.fit(X_train, y_train)
    f1v = f1_score(y_val, clf.predict(X_val), average='macro')
    print(f"TF-IDF baseline C={C}: Macro-F1 val={f1v:.4f}")
    if f1v > best_f1v:
        best_f1v, best_C = f1v, C

X_trainval = np.vstack([X_train, X_val])
y_trainval = list(y_train) + list(y_val)
clf_final = LogisticRegression(max_iter=1000, class_weight='balanced', C=best_C)
clf_final.fit(X_trainval, y_trainval)
y_pred = clf_final.predict(X_test)
f1t = f1_score(y_test, y_pred, average='macro')
print(f"\n=== BASELINE TF-IDF (C={best_C}, meme protocole train+val -> test) ===")
print(f"Macro-F1 test = {f1t:.4f}")
print(classification_report(y_test, y_pred, digits=3))
