"""
Filtre de cohérence géographique/thématique appliqué UNIQUEMENT à l'affichage
des sources, jamais au seuil anti-hallucination (0.20) ni à la classification
(voir main.py: ce module n'est appelé qu'après que le verdict est décidé).

Principe : un petit lexique de ~35 entités géographiques (pays/régions/villes
pertinentes pour le climat en Afrique de l'Ouest et au-delà) et thématiques
(phénomènes climatiques) est comparé entre le claim et chaque evidence
retournée par FAISS. Si le claim contient au moins une entité reconnaissable
et qu'aucune ne se retrouve dans l'evidence, la source est marquée comme
"pertinence incertaine" (elle reste affichée - voir DOCUMENTATION_TECHNIQUE.md
pour la justification de ce choix plutôt que de la masquer).

Limites assumées (documentées) : couverture lexicale forcément incomplète
(toutes les villes/phénomènes ne sont pas listés) ; un claim qui ne contient
aucune entité reconnaissable par ce lexique n'est jamais filtré, par choix
délibéré pour ne pas masquer de l'information par excès de prudence.
"""
import re
import unicodedata

# Chaque entrée : (identifiant canonique, [formes de surface à détecter]).
# Formes de surface données sans accents/apostrophes : la normalisation du
# texte (voir normalize()) retire accents et apostrophes avant comparaison,
# donc "Côte d'Ivoire" et "Ivory Coast" sont bien deux formes du même groupe.
GEO_ENTITIES = [
    ("cote_ivoire", ["cote d ivoire", "ivory coast", "ivoirien", "ivoirienne"]),
    ("abidjan", ["abidjan"]),
    ("cocody", ["cocody"]),
    ("bouake", ["bouake"]),
    ("korhogo", ["korhogo"]),
    ("yamoussoukro", ["yamoussoukro"]),
    ("grand_bassam", ["grand bassam"]),
    ("san_pedro", ["san pedro"]),
    ("afrique_ouest", ["afrique de l ouest", "west africa"]),
    ("afrique_subsaharienne", ["afrique subsaharienne", "sub saharan africa", "sub-saharan africa"]),
    ("senegal", ["senegal"]),
    ("dakar", ["dakar"]),
    ("somalie", ["somalie", "somalia"]),
    ("corne_afrique", ["corne de l afrique", "horn of africa"]),
    ("kenya", ["kenya"]),
    ("ghana", ["ghana"]),
    ("kumasi", ["kumasi"]),
    ("nigeria", ["nigeria"]),
    ("mali", ["mali"]),
    ("burkina_faso", ["burkina faso"]),
    ("guinee", ["guinee", "guinea"]),
    ("liberia", ["liberia"]),
    ("congo", ["congo", "rdc", "drc"]),
    ("goma", ["goma"]),
]

THEME_ENTITIES = [
    ("pluie", ["pluie", "pluviometrie", "rain", "precipitation", "precipitations"]),
    ("temperature", ["temperature", "temperatures"]),
    ("secheresse", ["secheresse", "drought"]),
    ("inondation", ["inondation", "inondations", "flood", "flooding"]),
    ("chaleur", ["chaleur", "vague de chaleur", "heat", "heatwave"]),
    ("niveau_mer", ["niveau de la mer", "sea level", "sea-level"]),
    ("rechauffement", ["rechauffement climatique", "changement climatique", "global warming", "climate change"]),
    ("paludisme", ["paludisme", "malaria"]),
    ("recolte", ["recolte", "harvest"]),
    ("cacao", ["cacao", "cocoa"]),
]

ALL_ENTITIES = GEO_ENTITIES + THEME_ENTITIES

# Une ville implique son pays (ex. une source nationale sur la Côte d'Ivoire
# est pertinente pour une question sur Abidjan, même si "Abidjan" n'est pas
# mot pour mot dans l'evidence) - sans quoi une source Banque Mondiale sur la
# Côte d'Ivoire serait marquée "incertaine" pour une question sur Cocody, ce
# qui serait un faux positif du filtre lui-même. Une seule strate
# ville -> pays est utilisée volontairement (pas de remontée pays -> région)
# pour rester simple et éviter de trop diluer la spécificité géographique.
CITY_TO_COUNTRY = {
    "abidjan": "cote_ivoire",
    "cocody": "cote_ivoire",
    "bouake": "cote_ivoire",
    "korhogo": "cote_ivoire",
    "yamoussoukro": "cote_ivoire",
    "grand_bassam": "cote_ivoire",
    "san_pedro": "cote_ivoire",
    "dakar": "senegal",
    "kumasi": "ghana",
    "goma": "congo",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[’'`]", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(text: str) -> set:
    """Retourne l'ensemble des identifiants canoniques (géo + thème) détectés dans `text`."""
    if not text:
        return set()
    normalized = _normalize(text)
    found = set()
    for canonical_id, surface_forms in ALL_ENTITIES:
        for form in surface_forms:
            if re.search(r"\b" + re.escape(form) + r"\b", normalized):
                found.add(canonical_id)
                break
    # Expansion ville -> pays (voir CITY_TO_COUNTRY).
    for city_id in list(found):
        if city_id in CITY_TO_COUNTRY:
            found.add(CITY_TO_COUNTRY[city_id])
    return found


def is_relevance_uncertain(claim: str, evidence: str) -> bool:
    """
    True si le claim contient au moins une entité géo/thème reconnaissable et
    qu'aucune ne se retrouve dans l'evidence (source probablement hors-sujet
    malgré un score cosinus au-dessus du seuil). False si le claim ne contient
    aucune entité reconnaissable (on ne filtre jamais dans ce cas) ou s'il y a
    un chevauchement.
    """
    claim_entities = extract_entities(claim)
    if not claim_entities:
        return False
    evidence_entities = extract_entities(evidence)
    return claim_entities.isdisjoint(evidence_entities)
