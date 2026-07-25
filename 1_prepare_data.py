import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os

print("Téléchargement du dataset Climate-FEVER...")
dataset = load_dataset("tdiggelm/climate_fever", split="test")

# Extract pairs (claim, evidence, label)
print("Extraction des paires Claim-Preuve...")
data = []
for item in dataset:
    claim_id = item['claim_id']
    claim = item['claim']
    claim_label = item['claim_label']
    
    # Map labels: 0 -> SUPPORTS, 1 -> REFUTES, 2 -> NOT_ENOUGH_INFO
    # On ignore DISPUTED (3) si on veut se limiter à 3 classes comme spécifié
    if claim_label == 0:
        label = "SUPPORTS"
    elif claim_label == 1:
        label = "REFUTES"
    elif claim_label == 2:
        label = "NOT_ENOUGH_INFO"
    else:
        continue
        
    for ev in item['evidences']:
        evidence_text = ev['evidence']
        data.append({
            "claim_id": claim_id,
            "claim": claim,
            "evidence": evidence_text,
            "label": label
        })

df = pd.DataFrame(data)
print(f"Total des paires extraites : {len(df)}")

# Split sans data leakage (par claim_id)
claim_ids = df['claim_id'].unique().tolist()
train_ids, test_val_ids = train_test_split(claim_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)

train_df = df[df['claim_id'].isin(train_ids)]
val_df = df[df['claim_id'].isin(val_ids)]
test_df = df[df['claim_id'].isin(test_ids)]

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# Corpus de preuves uniques pour FAISS
corpus_df = pd.DataFrame(df['evidence'].unique(), columns=['evidence'])
corpus_df.to_csv("data/corpus.csv", index=False)

print(f"Splits créés : Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})")
print(f"Corpus de preuves uniques : {len(corpus_df)}")
print("Terminé ! Les données sont dans le dossier 'data/'")
