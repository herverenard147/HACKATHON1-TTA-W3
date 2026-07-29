# Legacy — fichiers obsolètes, non utilisés en production

Ce dossier contient d'anciens prototypes conservés pour référence historique. **Aucun des deux n'est utilisé par l'application en production.** Le seul point d'entrée officiel est `main.py` à la racine du projet (`uvicorn main:app`).

## `api.py`

Première version de l'API FastAPI, remplacée par `main.py` (commit "Refonte TERRAVA-AI (React + FastAPI)").

⚠️ **Ne pas exécuter ce fichier en pensant obtenir le comportement documenté.** Différences notables par rapport à `main.py` :
- Seuil anti-hallucination fixé à **0.45** au lieu de **0.20** (celui décrit dans le cahier des charges et implémenté dans `main.py`).
- Un seul endpoint `/verify`, `k=1` (pas de top-3 sources), pas de middleware CORS, pas d'endpoint d'upload PDF.

## `app.py`

Prototype d'interface **Streamlit** ("ClimaCheck Pro"), antérieur au passage à l'architecture React + FastAPI actuelle. Nécessite le package `streamlit`, volontairement retiré de `requirements.txt` racine car inutilisé par l'application active — installez-le manuellement (`pip install streamlit`) si vous voulez relancer ce prototype.

---

*Conservés uniquement à titre d'historique de conception. Pour toute évolution du produit, partir de `main.py` + `/frontend`.*
