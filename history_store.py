"""
Stockage de l'historique personnalisé des vérifications (phase 3.2).

Choix d'architecture : SQLite local (fichier history.db à la racine du
dépôt, gitignoré comme les autres artefacts générés à l'exécution -
models_saved/). Aucune base de données n'existait déjà dans le projet
(seulement des CSV statiques pour le corpus et les jeux d'entraînement) ;
SQLite est la solution la plus simple et cohérente avec une app FastAPI
mono-instance qui tourne déjà en local, sans ajouter d'infrastructure
(pas de serveur DB séparé à démarrer/administrer).

Identifiant utilisateur léger : une chaîne libre (`user_id`) générée et
conservée côté client (voir frontend/src/userId.ts), SANS mot de passe ni
vérification d'identité. Quiconque connaît un `user_id` peut consulter
l'historique associé - ce n'est PAS un mécanisme de sécurité, seulement un
identifiant de commodité pour retrouver ses propres vérifications sur le
même navigateur. Voir DOCUMENTATION_TECHNIQUE.md pour les limites assumées.
"""
import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import List

DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                claim TEXT NOT NULL,
                comprehension_level TEXT NOT NULL,
                badge_class TEXT NOT NULL,
                badge_icon TEXT NOT NULL,
                badge_text TEXT NOT NULL,
                analyse_text TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_verifications_user_id ON verifications(user_id)")
        conn.commit()
    finally:
        conn.close()


def save_verification(user_id: str, claim: str, comprehension_level: str,
                       badge_class: str, badge_icon: str, badge_text: str,
                       analyse_text: str, sources: list) -> int:
    """Enregistre une vérification TELLE QU'ELLE A ÉTÉ RENDUE à l'utilisateur
    (verdict, texte de niveau, sources déjà décidés par check_claim) - ne
    recalcule jamais rien. Retourne l'id de l'entrée créée."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO verifications
               (user_id, claim, comprehension_level, badge_class, badge_icon,
                badge_text, analyse_text, sources_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, claim, comprehension_level, badge_class, badge_icon,
             badge_text, analyse_text, json.dumps(sources, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_history(user_id: str) -> List[dict]:
    """Retourne UNIQUEMENT les vérifications de ce user_id, les plus
    récentes en premier. Aucune fuite vers un autre user_id : filtrage
    strict par égalité en SQL (WHERE user_id = ?)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM verifications WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        return [
            {
                "id": row["id"],
                "claim": row["claim"],
                "comprehension_level": row["comprehension_level"],
                "badge_class": row["badge_class"],
                "badge_icon": row["badge_icon"],
                "badge_text": row["badge_text"],
                "analyse_text": row["analyse_text"],
                "sources": json.loads(row["sources_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


