"""
Script de veille légère (phase 7) : vérifie si de nouveaux rapports/
publications GIEC (IPCC) ou OMM (WMO) sont apparus, et SIGNALE la
détection (stdout + fichier de rapport dans veille_reports/) sans ingérer
automatiquement dans le corpus FAISS. L'ingestion reste un geste manuel
(voir 4_ingest_documents.py / update_corpus.py) - ce script s'arrête
volontairement à la détection, comme périmètre annoncé pour cette phase.

Sources surveillées (vérifiées accessibles avant écriture de ce script,
pas devinées) :
- GIEC/IPCC : flux RSS officiel https://www.ipcc.ch/feed/ (confirmé
  fonctionnel : HTTP 200, Content-Type application/rss+xml).
- OMM/WMO : recherche d'un flux RSS public infructueuse sur wmo.int
  (aucun <link rel="alternate" type="application/rss+xml"> exposé sur la
  page d'accueil ni sur les chemins usuels testés) - repli sur une
  extraction des liens de la page de publications
  https://wmo.int/resources/publications (confirmée HTTP 200, liste bien
  des publications réelles, ex. "State of Climate in Africa").

État conservé entre deux exécutions dans veille_state.json (racine du
dépôt, gitignoré comme history.db) : liste des identifiants (URLs) déjà
vus par source. Au tout premier lancement, tout est "nouveau" par
définition ; les lancements suivants ne signalent que les entrées absentes
de l'état précédent.

Lancement manuel :
    source venv/bin/activate
    python3 scripts/veille_giec_omm.py

Planification (à la charge de l'utilisateur - non automatisée dans cette
session, volontairement laissée manuelle/documentée) : exemple de tâche
cron pour une vérification quotidienne à 8h -
    0 8 * * * cd /chemin/vers/terrava-ai && venv/bin/python3 scripts/veille_giec_omm.py >> veille.log 2>&1

Ce que ce script fait : détecte et signale (log + fichier de rapport).
Ce que ce script NE fait PAS : télécharger le contenu des nouveaux
documents, les ajouter au corpus, ou reconstruire l'index FAISS - une
ingestion complète automatisée reste une perspective documentée
(DOCUMENTATION_TECHNIQUE.md, section pistes futures), pas implémentée ici.
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

import requests
import feedparser

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(ROOT_DIR, "veille_state.json")
REPORT_DIR = os.path.join(ROOT_DIR, "veille_reports")

SOURCES = [
    {
        "name": "GIEC (IPCC)",
        "type": "rss",
        "url": "https://www.ipcc.ch/feed/",
    },
    {
        "name": "OMM (WMO) - Publications",
        "type": "html_links",
        "url": "https://wmo.int/resources/publications",
        "link_pattern": r'href="(/resources/publication[^"]*)"',
        "base_url": "https://wmo.int",
    },
]


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def check_rss_source(source: dict) -> list:
    feed = feedparser.parse(source["url"])
    return [{"id": entry.link, "title": entry.title, "link": entry.link} for entry in feed.entries]


def check_html_links_source(source: dict) -> list:
    resp = requests.get(source["url"], timeout=15)
    resp.raise_for_status()
    links = sorted(set(re.findall(source["link_pattern"], resp.text)))
    items = []
    for link in links:
        full_url = source["base_url"] + link
        title = link.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
        items.append({"id": full_url, "title": title, "link": full_url})
    return items


def run_veille() -> bool:
    """Retourne True si au moins une nouveauté a été détectée sur une source."""
    state = load_state()
    report_lines = [f"=== Rapport de veille GIEC/OMM — {datetime.now(timezone.utc).isoformat()} ==="]
    any_new = False

    for source in SOURCES:
        name = source["name"]
        seen_ids = set(state.get(name, []))
        try:
            if source["type"] == "rss":
                items = check_rss_source(source)
            else:
                items = check_html_links_source(source)
        except Exception as e:
            report_lines.append(f"[{name}] ÉCHEC de la vérification : {e}")
            continue

        new_items = [item for item in items if item["id"] not in seen_ids]
        if new_items:
            any_new = True
            report_lines.append(f"[{name}] {len(new_items)} nouvelle(s) entrée(s) détectée(s) :")
            for item in new_items:
                report_lines.append(f"  - {item['title']} ({item['link']})")
        else:
            report_lines.append(f"[{name}] Aucune nouveauté ({len(items)} entrée(s) déjà connue(s)).")

        state[name] = sorted(seen_ids | {item["id"] for item in items})

    save_state(state)

    report_text = "\n".join(report_lines)
    print(report_text)

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_filename = os.path.join(
        REPORT_DIR, f"veille_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.txt"
    )
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    return any_new


if __name__ == "__main__":
    run_veille()
    sys.exit(0)
