import pandas as pd
import os

corpus_file = "data/corpus.csv"
print(f"Migration de {corpus_file}...")

if os.path.exists(corpus_file):
    df = pd.read_csv(corpus_file)
    if 'institution' not in df.columns:
        df['institution'] = "Climate-FEVER (Wikipedia)"
        df['title'] = "Jeu de données initial"
        df['year'] = "2020"
        df['url'] = "https://huggingface.co/datasets/tdiggelm/climate_fever"
        df.to_csv(corpus_file, index=False)
        print("Migration terminée avec succès. Colonnes ajoutées.")
    else:
        print("Le fichier contient déjà les colonnes de métadonnées.")
else:
    print("Fichier corpus.csv introuvable.")
