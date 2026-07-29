 🌍 TERRAVA-AI : Plateforme d'Intelligence et de Fact-Checking Climatique

**TERRAVA-AI** (anciennement ClimaCheck) est un outil de vérification des faits (fact-checking) propulsé par l'Intelligence Artificielle. Conçu spécifiquement pour les journalistes, chercheurs et décideurs climatiques, il permet de confronter instantanément une déclaration aux données officielles de la littérature scientifique (GIEC, OMM, Banque Mondiale) afin de lutter contre la désinformation climatique.


  Fonctionnalités Principales

- **Détection Anti-Désinformation :** Évalue si une affirmation est `CONFIRMÉE`, `RÉFUTÉE` ou `NON VÉRIFIABLE` par la science.
- **Architecture Zéro-GPU :** Modèle hybride ultra-optimisé combinant une base vectorielle (FAISS) et un classifieur de Machine Learning (Régression Logistique) capable de tourner sur un simple ordinateur CPU local.
- **Traçabilité Totale :** Les sources institutionnelles exactes ayant servi à la décision sont toujours affichées à l'utilisateur (Citations, Liens, Années).
- **Analyse de Documents (PDF) :** Importez un document par Glisser-Déposer pour extraire instantanément le texte et lancer l'analyse.
- **Filtre Régional :** Sélecteur de zone géographique (Global / Afrique de l'Ouest / Côte d'Ivoire) côté interface. Le corpus contient un focus institutionnel sur cette région (GIEC, OMM, Banque Mondiale), mais **le filtre n'est pas encore appliqué côté backend** : `zone_geo` est transmis à l'API mais actuellement ignoré par `check_claim()` dans `main.py` — la recherche FAISS reste globale quelle que soit la zone sélectionnée. *(Limitation connue, pas un bug caché : à implémenter si le filtrage réel par région est souhaité.)*


 Architecture Technique (SaaS)

Le projet a été refondu pour adopter un standard industriel **Full-Stack** :

 1. Le "Cerveau" : Back-End (Python / FastAPI)
L'API REST est exposée via `main.py` et orchestre :
- L'encodeur de similarité sémantique (`all-MiniLM-L6-v2`).
- La base de connaissances vectorielle (`FAISS`).
- L'algorithme de logique de vérité scientifique (`Joblib / Régression Logistique scikit-learn`, `class_weight='balanced'`, `max_iter=1000`).
- Le parseur natif de PDF (`PyPDF2`).

### 2. Le "Visage" : Front-End (React / Tailwind CSS)
Une interface moderne "Scientific Workbench" gérée dans le dossier `/frontend` :
- Créée avec **Vite + React + TypeScript**.
- Composants stylisés sur-mesure via **Tailwind CSS**.
- Design épuré, accessible et totalement exempt de jargon technique.


 Guide d'Installation et d'Exécution

 Prérequis
- **Python 3.9+** (Pour l'API IA)
- **Node.js 18+** (Pour l'interface React)

 Étape 1 : Lancer le Serveur IA (Back-End)

```bash
# Dans le dossier principal du projet :
pip install -r requirements.txt

# Démarrer l'API sur le port 8000
uvicorn main:app --host 127.0.0.1 --port 8000
```
*L'API est désormais disponible sur `http://localhost:8000/docs`.*

### Étape 2 : Lancer l'Interface Graphique (Front-End)

```bash
# Dans un NOUVEAU terminal, se rendre dans le dossier frontend :
cd frontend

# Installer les dépendances
npm install

# Lancer le serveur de développement React
npm run dev
```
*Le portail TERRAVA-AI s'ouvrira sur `http://localhost:5173`.*

---

 Les Scripts de Modélisation (MLOps)
Ce dépôt inclut également les scripts ayant servi à la conception de l'IA (idéal pour la mise à jour des rapports). **L'ordre d'exécution ci-dessous est important** :

| Ordre | Script | Rôle |
|---|---|---|
| 1 | `1_prepare_data.py` | Télécharge le dataset Climate-FEVER (HuggingFace `datasets`) et génère `data/train.csv`, `data/val.csv`, `data/test.csv`, ainsi qu'un `data/corpus.csv` **brut** (une seule colonne `evidence`, sans métadonnées). |
| 2 | `migrate_csv.py` | Ajoute les colonnes `institution` / `title` / `year` / `url` au corpus brut. **Étape obligatoire avant l'étape 4** — sans elle, les preuves issues de Climate-FEVER s'affichent sans source exploitable. |
| 3 | `2_build_retrieval.py` | Encode `data/corpus.csv` avec `all-MiniLM-L6-v2` et construit l'index vectoriel FAISS (`models_saved/faiss_index.bin`). |
| 4 | `4_ingest_documents.py` | Ajoute les documents institutionnels (GIEC AR6, OMM, Banque Mondiale) au corpus avec leurs métadonnées, puis reconstruit l'index FAISS avec l'ensemble enrichi. |
| 5 *(optionnel)* | `update_corpus.py` | Fusionne `data/corpus_additionnel.csv` (affirmations régionales additionnelles) dans le corpus et reconstruit l'index FAISS. |
| — | `3_train_classifier.py` | Entraîne la Régression Logistique (`class_weight='balanced'`, `max_iter=1000`) sur `train.csv` / `test.csv` et sauvegarde `models_saved/classifier.joblib`. Indépendant du corpus/FAISS : peut être lancé à tout moment après l'étape 1. |

> ⚠️ **Attention — ne pas casser le corpus enrichi.** `data/corpus.csv` est déjà fourni dans ce dépôt en version enrichie (Climate-FEVER **+** documents institutionnels **+** métadonnées `institution/title/year/url`). **Ne relance pas `1_prepare_data.py` isolément** : ce script écrase `data/corpus.csv` par une version brute sans métadonnées ni documents institutionnels. Si tu dois régénérer les données depuis zéro, exécute bien la séquence complète `1 → migrate_csv → 2 → 4 → (5)` pour reconstruire un corpus équivalent, sans quoi les sources GIEC/OMM/Banque Mondiale disparaîtront de l'application et les preuves Climate-FEVER afficheront une institution manquante.

---
*Ce projet a été développé dans le cadre du Hackathon "TTA W3" pour proposer une solution frugale (Zéro-GPU) et à fort impact dans la lutte contre la désinformation climatique.*
