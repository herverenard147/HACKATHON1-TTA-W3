# TERRAVA-AI — Documentation technique

Ce document décrit l'état **réel et vérifié** du projet (code exécuté, métriques mesurées sur le jeu de test), pas les objectifs du cahier des charges initial. Chaque chiffre cité ici provient d'une exécution effective des scripts de ce dépôt.

---

## 1. Architecture générale

```
┌──────────────┐     ┌────────────────────┐     ┌─────────────────────────┐     ┌──────────────────┐
│   Entrée     │     │    Récupération     │     │  Feature engineering    │     │  Classification   │
│   (claim)    │ --> │  (encodage + FAISS  │ --> │  NLI (concat 4×384)     │ --> │  + logique verdict │
│  texte libre │     │   top-k=3)          │     │                         │     │                    │
└──────────────┘     └────────────────────┘     └─────────────────────────┘     └──────────────────┘
                                                                                          │
                                                                                          v
                                                                                 ┌──────────────────┐
                                                                                 │  Verdict sourcé   │
                                                                                 │  affiché (React)  │
                                                                                 └──────────────────┘
```

Étapes détaillées (voir `main.py`, fonction `check_claim`) :

1. **Entrée** : une affirmation climatique en texte libre (français ou anglais), reçue par `POST /api/check-claim`.
2. **Récupération** : le claim est encodé en vecteur de 384 dimensions (`all-MiniLM-L6-v2`, normalisé), puis comparé aux ~4870 preuves du corpus via un index FAISS `IndexFlatIP` (produit scalaire = similarité cosinus car les vecteurs sont normalisés). Les 3 preuves les plus proches (top-k=3) sont retenues.
3. **Garde-fou anti-hallucination** : si le score cosinus de la preuve la plus proche (top-1) est **< 0.20**, le pipeline s'arrête ici et renvoie directement `NON VÉRIFIABLE`, sans appeler le classificateur.
4. **Feature engineering NLI** : sinon, l'evidence top-1 est encodée à son tour, et un vecteur de 1536 dimensions est construit par concaténation de 4 blocs de 384 dimensions chacun (détail section 3).
5. **Classification** : une Régression Logistique scikit-learn prédit une classe parmi `SUPPORTS` / `REFUTES` / `NOT_ENOUGH_INFO`.
6. **Verdict final** : traduction de la classe en verdict affiché (section 4), avec les 3 sources top-k citées (institution, extrait, titre, année, lien).

---

## 2. Stack technique et choix

| Composant | Choix | Pourquoi |
|---|---|---|
| Backend | **FastAPI** (async) | API REST légère, documentation OpenAPI automatique (`/docs`), validation de schéma via Pydantic, adaptée à un service CPU mono-modèle sans besoin de queue/worker complexe. |
| Recherche vectorielle | **FAISS `IndexFlatIP`** | Recherche exacte (pas d'approximation ANN) adaptée à un corpus de taille modeste (~4870 vecteurs) : latence négligeable, aucun compromis de rappel. `IndexFlatIP` (produit scalaire) sur vecteurs normalisés = équivalent exact de la similarité cosinus. |
| Encodeur sémantique | **SentenceTransformers `all-MiniLM-L6-v2`** | 22M paramètres, 384 dimensions, tourne en quelques millisecondes par phrase sur CPU. Compromis qualité/vitesse standard pour de la recherche sémantique légère, pas de dépendance GPU. |
| Classificateur | **Régression Logistique scikit-learn** (`class_weight='balanced'`, `C=0.1`, `max_iter=1000`) | Modèle linéaire interprétable, entraînement en secondes sur CPU, pas de risque d'overfitting catastrophique avec une régularisation adaptée (voir section 6 sur le choix de `C`). |
| Frontend | **React + Vite + TypeScript + Tailwind CSS** | SPA légère, hot-reload rapide en développement, typage statique pour réduire les erreurs d'intégration avec l'API. |
| Parsing PDF | **PyPDF2** | Extraction de texte suffisante pour préremplir le champ de saisie depuis un document déposé par l'utilisateur ; pas besoin d'OCR pour ce cas d'usage. |

---

## 3. Feature engineering NLI (détail)

Le vecteur d'entrée du classificateur est construit ainsi (`create_features()` dans `3_train_classifier.py`, logique identique dans `main.py`) :

```
c_emb  = encode(claim)      # 384 dims, normalisé
e_emb  = encode(evidence)   # 384 dims, normalisé

abs_diff          = |c_emb − e_emb|        # 384 dims
elementwise_mult  = c_emb ⊙ e_emb          # 384 dims

features = concat(c_emb, e_emb, abs_diff, elementwise_mult)   # 1536 dims
```

Interprétation :
- `c_emb`, `e_emb` : information brute des deux textes.
- `abs_diff` : capture la divergence terme-à-terme entre les deux représentations (utile pour détecter une contradiction directionnelle).
- `elementwise_mult` : capture le chevauchement / l'alignement sémantique (proche de la similarité cosinus mais dimension par dimension, avant agrégation).

C'est un pattern classique de feature engineering pour la classification de paires de phrases (NLI), permettant à un modèle linéaire de capter des interactions que la seule concaténation de `c_emb`/`e_emb` ne capturerait pas.

---

## 4. Seuil anti-hallucination et logique de décision

```python
if similarity_score < 0.20:
    verdict = "NON_VERIFIABLE"
else:
    raw_verdict = classifier.predict(features)[0]
    verdict = {"SUPPORTS": "CONFIRME", "REFUTES": "REFUTE"}.get(raw_verdict, "NON_VERIFIABLE")
```

- **Score < 0.20** : aucune preuve suffisamment proche sémantiquement n'existe dans le corpus → `NON VÉRIFIABLE`, badge "Aucune preuve scientifique", **aucune source affichée**. Le classificateur n'est même pas appelé.
- **Score ≥ 0.20** :
  - `SUPPORTS` → **CONFIRMÉ**
  - `REFUTES` → **RÉFUTÉ**
  - `NOT_ENOUGH_INFO` → **NON VÉRIFIABLE** (badge "Preuves indirectes/insuffisantes", sources affichées car le sujet est proche mais non tranché)

Ce seuil a été vérifié expérimentalement en phase 1 : une phrase en coréen hors-sujet obtient un score de 0.1987 (juste sous le seuil) et déclenche bien le verdict `NON VÉRIFIABLE` avec zéro source.

---

## 5. Pipeline de données (ordre d'exécution)

⚠️ **L'ordre ci-dessous est important.** `data/corpus.csv` est déjà fourni dans ce dépôt en version enrichie (Climate-FEVER + documents institutionnels + métadonnées). Relancer `1_prepare_data.py` seul **écrase** cette version enrichie par une version brute sans métadonnées ni documents institutionnels.

| Ordre | Script | Rôle |
|---|---|---|
| 1 | `1_prepare_data.py` | Télécharge Climate-FEVER (HuggingFace `datasets`) et génère `train.csv`/`val.csv`/`test.csv` + un `corpus.csv` **brut** (colonne `evidence` seule). |
| 2 | `migrate_csv.py` | Ajoute les colonnes `institution`/`title`/`year`/`url` au corpus brut. **Obligatoire avant l'étape 4.** |
| 3 | `2_build_retrieval.py` | Encode `corpus.csv` et construit l'index FAISS (`models_saved/faiss_index.bin`). |
| 4 | `4_ingest_documents.py` | Ajoute les documents institutionnels (GIEC, OMM, Banque Mondiale) au corpus avec métadonnées, reconstruit l'index FAISS. |
| 5 *(optionnel)* | `update_corpus.py` | Fusionne `corpus_additionnel.csv` (affirmations régionales additionnelles) et reconstruit l'index FAISS. |
| — | `3_train_classifier.py` | Entraîne la Régression Logistique sur `train.csv` + `val.csv`, évalue une seule fois sur `test.csv`. Indépendant du corpus/FAISS. |

---

## 6. Métriques réelles mesurées

**Méthodologie** : `data/val.csv` (1035 exemples) sert exclusivement à la sélection d'hyperparamètre (`C`), jamais à l'entraînement final ni à l'évaluation. `data/test.csv` (1040 exemples) n'est utilisé qu'**une seule fois**, en évaluation finale, jamais pendant le tuning. Le modèle final est réentraîné sur `train.csv` + `val.csv` après sélection de `C`, puis évalué sur `test.csv`.

### Résultat final (modèle en production, `models_saved/classifier.joblib`)

| Métrique | Valeur mesurée |
|---|---|
| **Macro-F1** | **0.532** |
| F1 — SUPPORTS | 0.638 |
| F1 — NOT_ENOUGH_INFO | 0.518 |
| F1 — REFUTES | 0.440 |
| Accuracy | 0.562 |

### Historique des itérations (mesurées, pas estimées)

| Configuration | Macro-F1 test |
|---|---|
| Baseline initiale (`C=1.0` défaut, entraîné sur train seul) | 0.482 |
| `C=0.1` (grid search sur val), entraîné sur train seul | 0.516 |
| `C=0.1`, entraîné sur train+val (config finale déployée) | **0.532** |
| Calibration Platt (`CalibratedClassifierCV`, sigmoid) sur `C=0.1` | 0.461 *(rejetée — dégrade)* |
| **Baseline de comparaison : TF-IDF** (mêmes 4 blocs de features, mêmes réglages, `max_features=2000`) | 0.485 |

La baseline TF-IDF (vectorisation lexicale classique, sans embeddings sémantiques) atteint 0.485 avec le même pipeline de classification — l'encodeur `all-MiniLM-L6-v2` apporte un gain réel mais modeste (+0.047 absolu) par rapport à une approche purement lexicale, sur ce jeu de données.

### Ce que ce chiffre signifie concrètement

Sur les 1040 paires claim-evidence du jeu de test (jamais vues pendant l'entraînement), le système donne le bon verdict (SUPPORTS/REFUTES/NOT_ENOUGH_INFO) dans **56.2% des cas** (accuracy), avec une performance très inégale selon la classe : bonne sur SUPPORTS, moyenne sur NOT_ENOUGH_INFO, faible sur REFUTES (le système confond souvent une affirmation qui *contredit* une preuve avec une affirmation qu'elle *confirme*, quand les deux portent sur le même sujet).

### Pourquoi le corpus institutionnel n'explique pas l'écart

Le corpus institutionnel (GIEC, OMM, Banque Mondiale) ne représente que **3 chunks sur ~4870** dans `data/corpus.csv`, et **aucun** n'apparaît dans `train.csv`/`val.csv`/`test.csv` — ces fichiers proviennent à 100% de Climate-FEVER (Wikipedia). Le corpus institutionnel sert uniquement à la récupération FAISS en production, jamais à l'entraînement ou à l'évaluation du classificateur. 100% de l'écart de Macro-F1 mesuré est donc attribuable à Climate-FEVER seul.

### Pistes testées et écartées

- **Calibration Platt** (`CalibratedClassifierCV`, sigmoid, cv=5) : dégrade le Macro-F1 test (0.516 → 0.461 sur la config train-seul). Améliore la qualité des probabilités mais lisse les frontières de décision au détriment de l'exactitude des labels durs.
- **Rééquilibrage manuel des classes** au-delà de `class_weight='balanced'` : aucune configuration testée (grille de multiplicateurs sur REFUTES/NOT_ENOUGH_INFO) ne bat `'balanced'` natif sur validation.
- **Repondération post-hoc du seuil de décision** (argmax pondéré par classe, tuné sur validation) : gain apparent sur validation (0.5105 → 0.5247) qui **ne se généralise pas** sur test (0.5090 → 0.5020, en régression) — signe de surapprentissage à un set de validation de taille limitée (1035 exemples). Piste écartée pour cette raison précise.

---

## 7. Installation et lancement

### Prérequis
- Python 3.9+ (testé avec 3.12.3)
- Node.js 18+

### Backend

```bash
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

⚠️ **Le port 8000 peut être déjà occupé par un autre service sur votre machine** (observé en conditions réelles : un autre projet local répondant sur ce port). Si `uvicorn` échoue avec `address already in use`, relancez-le sur un autre port (`--port 8001` par exemple).

### Frontend

```bash
cd frontend
npm install

# Si le backend ne tourne pas sur le port 8000 par défaut :
cp .env.example .env
# puis éditez frontend/.env : VITE_API_BASE_URL=http://localhost:<votre_port>

npm run dev
```

L'URL du backend appelée par le frontend est centralisée dans `frontend/src/config.ts` et lue depuis la variable d'environnement `VITE_API_BASE_URL` (fichier `frontend/.env`, non commité). **Ne jamais coder cette URL en dur** dans le code source — c'était un bug corrigé après un incident réel où le frontend continuait d'appeler le port 8000 (occupé par un autre projet) alors que le backend tournait sur 8001.

### Régénérer les modèles depuis zéro

Voir la section 5 pour l'ordre exact. `models_saved/` (index FAISS + classificateur) n'est pas versionné (`.gitignore`) : ces artefacts doivent être régénérés localement via `2_build_retrieval.py` et `3_train_classifier.py` après un clone.

---

## 8. Limitations techniques connues

- **Corpus institutionnel quasi-absent de l'entraînement** : 3 chunks sur ~4870 dans le corpus de récupération (GIEC, OMM, Banque Mondiale), 0 dans les données labellisées train/val/test. L'apport réel du focus régional Afrique de l'Ouest/Côte d'Ivoire au *raisonnement* du classificateur est donc nul — il n'agit qu'au niveau de la récupération de sources à afficher.
- **Filtre régional cosmétique** : le sélecteur de zone géographique (`zone_geo`) de l'interface est transmis à l'API mais n'est pas exploité par `check_claim()` — la recherche FAISS reste globale quelle que soit la zone sélectionnée.
- **Lien d'archive limité aux documents connus** : les 2 documents institutionnels locaux (Banque Mondiale, OMM) sont désormais servis via `/documents/<fichier>` (backend, `StaticFiles`). Le document GIEC pointe vers une URL externe réelle (`ipcc.ch`) mais le fichier local correspondant (`data/climate_docs/GIEC_AR6_Afrique_Resume.pdf`) est en réalité un texte de simulation (limitation déjà documentée : le téléchargement réel du PDF n'a jamais été effectué par `4_ingest_documents.py`, qui écrit un texte de substitution avec l'extension `.pdf`) — cliquer sur ce lien externe mène au vrai site du GIEC, pas nécessairement à la page exacte citée.
- **Sources thématiquement/géographiquement non pertinentes possibles** : le seuil de 0.20 filtre les cas hors-sujet extrêmes, mais entre 0.20 et ~0.55 le système peut afficher une source qui partage un vocabulaire climatique générique avec le claim sans rapport géographique ou thématique réel avec lui. Exemple mesuré : pour une question portant sur une hausse localisée des précipitations dans un quartier d'Abidjan sur une période de deux ans, une des sources renvoyées porte sur la sécheresse dans la Corne de l'Afrique (score cosinus mesuré : 0.32, au-dessus du seuil, mais sans lien géographique — pays différent — ni thématique — sécheresse contre excès de pluie — avec la question). Ce n'est pas un cas isolé : sur 8 affirmations hyper-locales testées (villes différentes, phénomènes climatiques différents), aucune ne tombe sous 0.20, et certaines remontent même des extraits Climate-FEVER totalement hors sujet climatique (un extrait sur une pièce de théâtre classique a été retourné avec un score de 0.42 pour une question sur le paludisme lié à la chaleur). Cette limite tient à la nature du filtrage par similarité d'embeddings : la proximité sémantique globale (vocabulaire climatique commun) ne garantit pas la pertinence contextuelle précise (même lieu, même phénomène). Aucune vérification d'entité (lieu, sujet) n'est appliquée aujourd'hui avant l'affichage des sources — piste d'amélioration détaillée en section 9.
- **Granularité du corpus** : GIEC/OMM/Banque Mondiale documentent des tendances macro (nationales, régionales, mondiales, sur plusieurs décennies). Le système ne peut structurellement pas confirmer ou infirmer une statistique hyper-locale et récente (ex. "+80% de pluie en 2 ans dans un quartier précis") — ce type d'affirmation, pourtant fréquent dans la désinformation climatique qui circule sur les réseaux sociaux en Afrique de l'Ouest, tombe presque systématiquement en `NON VÉRIFIABLE` par manque de preuve directe, ce qui est le comportement correct mais peut donner l'impression d'un système peu utile sur ce type de cas précis.
- **Pas de calibration efficace** : testée (Platt scaling) et écartée car elle dégrade le Macro-F1 (section 6). Les scores de confiance affichés ne sont donc pas de vraies probabilités calibrées.
- **Pas de support multilingue structuré** : le corpus mélange français (documents institutionnels, `corpus_additionnel.csv`) et anglais (Climate-FEVER). `all-MiniLM-L6-v2` gère raisonnablement les deux langues en pratique mais aucune stratégie de traduction ou d'alignement cross-lingue n'est mise en œuvre — la qualité de la récupération pour une langue non représentée dans le corpus n'est pas garantie.
- **Classe REFUTES la plus faible** (F1=0.44) : voir section 6, confusion fréquente avec SUPPORTS quand claim et evidence portent sur le même sujet mais avec un signe opposé — limite connue d'un classificateur basé uniquement sur la proximité d'embeddings, sans modélisation explicite de la négation.
- **Corpus FAISS statique** : aucune ingestion continue ; toute mise à jour nécessite de relancer manuellement le pipeline (section 5).
- **Dépendances non pinnées dans les fichiers `legacy/`** : `api.py`/`app.py` (prototypes obsolètes) ne font pas partie du chemin de production et ne sont pas couverts par les garanties ci-dessus (voir `legacy/README.md`).

---

## 9. Pistes d'amélioration futures

- **Contrôle de cohérence géographique/thématique à l'affichage** *(proposé, non implémenté — en attente de validation)* : avant d'afficher une source dont le score dépasse 0.20, vérifier la présence d'une incohérence explicite entre entités nommées du claim et de l'evidence (ex. un lexique de ~30-50 noms de pays/villes d'Afrique de l'Ouest + mots-clés de phénomènes climatiques) pour masquer les sources clairement hors-sujet sans toucher au seuil ni à la classification. Gain estimé : réduit le bruit perçu sur les questions hyper-locales sans risque de faux négatif supplémentaire (n'affecte que l'affichage). Risque : couverture nécessairement incomplète du lexique, effort de maintenance, ne résout pas le cas où le claim ne contient aucune entité nommée reconnaissable.
- **Élargir le corpus institutionnel** : au-delà des 3 chunks actuels, ingérer davantage de rapports GIEC/OMM/Banque Mondiale complets (pas de simulation de téléchargement) pour que le focus régional annoncé ait un effet réel sur la couverture des sources.
- **Implémenter le filtre régional** (`zone_geo`) côté backend, par exemple en repondérant ou restreignant la recherche FAISS aux documents tagués pour la zone sélectionnée.
- **Explorer un jeu de données REFUTES plus riche** ou des techniques d'augmentation ciblées sur cette classe, seule à rester sous 0.5 de F1.
- **Ingestion continue du corpus** plutôt qu'un index FAISS statique reconstruit manuellement.
- **Stratégie multilingue explicite** (traduction automatique du claim vers la langue dominante du corpus avant recherche, ou corpus francophone étoffé) plutôt que de compter sur la robustesse cross-lingue implicite de l'encodeur.
