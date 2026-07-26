 🌍 TERRAVA-AI : Plateforme d'Intelligence et de Fact-Checking Climatique

**TERRAVA-AI** (anciennement ClimaCheck) est un outil de vérification des faits (fact-checking) propulsé par l'Intelligence Artificielle. Conçu spécifiquement pour les journalistes, chercheurs et décideurs climatiques, il permet de confronter instantanément une déclaration aux données officielles de la littérature scientifique (GIEC, OMM, Banque Mondiale) afin de lutter contre la désinformation climatique.


  Fonctionnalités Principales

- **Détection Anti-Désinformation :** Évalue si une affirmation est `CONFIRMÉE`, `RÉFUTÉE` ou `NON VÉRIFIABLE` par la science.
- **Architecture Zéro-GPU :** Modèle hybride ultra-optimisé combinant une base vectorielle (FAISS) et un classifieur de Machine Learning (Random Forest) capable de tourner sur un simple ordinateur CPU local.
- **Traçabilité Totale :** Les sources institutionnelles exactes ayant servi à la décision sont toujours affichées à l'utilisateur (Citations, Liens, Années).
- **Analyse de Documents (PDF) :** Importez un document par Glisser-Déposer pour extraire instantanément le texte et lancer l'analyse.
- **Filtre Régional :** Focus spécifique sur l'Afrique de l'Ouest et la Côte d'Ivoire.


 Architecture Technique (SaaS)

Le projet a été refondu pour adopter un standard industriel **Full-Stack** :

 1. Le "Cerveau" : Back-End (Python / FastAPI)
L'API REST est exposée via `main.py` et orchestre :
- L'encodeur de similarité sémantique (`all-MiniLM-L6-v2`).
- La base de connaissances vectorielle (`FAISS`).
- L'algorithme de logique de vérité scientifique (`Joblib / Random Forest`).
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
Ce dépôt inclut également les scripts ayant servi à la conception de l'IA (idéal pour la mise à jour des rapports) :
- `1_prepare_data.py` : Scrape et structure les données des rapports bruts.
- `2_build_retrieval.py` : Indexe les données dans la base vectorielle FAISS.
- `3_train_classifier.py` : Entraîne le modèle Random Forest sur la logique d'inclusion/contradiction.
- `4_ingest_documents.py` : Ajout de nouveaux PDFs (GIEC, BM) à la base.

---
*Ce projet a été développé dans le cadre du Hackathon "TTA W3" pour proposer une solution frugale (Zéro-GPU) et à fort impact dans la lutte contre la désinformation climatique.*
