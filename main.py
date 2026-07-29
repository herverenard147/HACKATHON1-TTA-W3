from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import joblib
import PyPDF2
import io
import os
from typing import List, Optional
from relevance_filter import is_relevance_uncertain, extract_entities

# OCR de repli pour les pages PDF sans couche texte (scan/image). Import
# optionnel : si pytesseract/pdf2image ou le binaire système tesseract-ocr
# ne sont pas installés, l'app démarre quand même — l'OCR est simplement
# indisponible et les pages sans texte natif restent vides comme avant
# (voir upload_pdf). Ne jamais bloquer tout le backend pour cette dépendance
# optionnelle système (documentée dans README.md).
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Poids additif appliqué au score cosinus d'une source candidate quand sa zone
# géographique (déduite du même lexique que relevance_filter.py) recoupe la
# zone demandée par l'utilisateur. Calibré empiriquement : les chunks
# institutionnels pertinents mais peu représentés dans le corpus (3 sur ~4870)
# se classent souvent autour de 0.10-0.15 sous les meilleurs chunks Climate-FEVER
# génériques (score cosinus) ; ce boost suffit à les faire remonter dans le
# top-k affiché sans écraser un score sémantique nettement supérieur.
ZONE_GEO_BOOST = 0.15

app = FastAPI(
    title="TERRAVA-AI API",
    description="API Back-End pour la vérification climatique",
    version="2.0"
)

# Configuration CORS pour autoriser React (Vite tourne souvent sur le port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En production, limiter aux domaines précis
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sert les documents institutionnels ingérés localement (data/climate_docs/) pour
# que le lien "Consulter l'archive" des sources sans URL externe pointe vers un
# contenu réel au lieu d'un placeholder "local" mal interprété par le frontend.
if os.path.isdir("data/climate_docs"):
    app.mount("/documents", StaticFiles(directory="data/climate_docs"), name="documents")

# Modèles de données
class ClaimRequest(BaseModel):
    claim: str
    zone_geo: str = "Global (International)"

class Source(BaseModel):
    institution: str
    evidence: str
    title: str
    year: str
    url: str
    # Filtre d'affichage uniquement (post-classification, n'affecte ni le
    # seuil de 0.20 ni le verdict) : True si le claim contient une entité
    # géographique/thématique reconnue qui ne se retrouve pas dans cette
    # evidence précise. Voir relevance_filter.py.
    relevance_uncertain: bool = False

class VerificationResponse(BaseModel):
    badge_class: str
    badge_icon: str
    badge_text: str
    analyse_text: str
    sources: List[Source]

# Variables globales pour l'IA
embedding_model = None
index = None
classifier = None
corpus_df = None

@app.on_event("startup")
def load_models():
    global embedding_model, index, classifier, corpus_df
    print("Démarrage du moteur TERRAVA-AI...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        index = faiss.read_index("models_saved/faiss_index.bin")
        classifier = joblib.load("models_saved/classifier.joblib")
        corpus_df = pd.read_csv("data/corpus.csv")
        print("Moteur d'IA chargé avec succès.")
    except Exception as e:
        print(f"Erreur fatale lors du chargement des modèles : {e}")

@app.post("/api/check-claim", response_model=VerificationResponse)
def check_claim(request: ClaimRequest):
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="La déclaration est vide.")
        
    try:
        c_emb = embedding_model.encode([request.claim], normalize_embeddings=True)
        k = 3
        distances, indices = index.search(c_emb, k)
        
        top_evidence_row = corpus_df.iloc[indices[0][0]]
        top_evidence = top_evidence_row['evidence']
        similarity_score = float(distances[0][0])
        
        # Filtre anti-hallucination (Seuil de tolérance à 0.20 comme décidé)
        if similarity_score < 0.20:
            verdict = "NON_VERIFIABLE"
        else:
            e_emb = embedding_model.encode([top_evidence], normalize_embeddings=True)
            features = np.hstack((c_emb, e_emb, np.abs(c_emb - e_emb), c_emb * e_emb))
            raw_verdict = classifier.predict(features)[0]
            
            if raw_verdict == "SUPPORTS":
                verdict = "CONFIRME"
            elif raw_verdict == "REFUTES":
                verdict = "REFUTE"
            else:
                verdict = "NON_VERIFIABLE"
                
        # Formatage de la réponse selon le verdict
        if verdict == "CONFIRME":
            badge_class = "badge-confirmed"
            badge_icon = "✅"
            badge_text = "CONFIRMÉ PAR LES DONNÉES SCIENTIFIQUES"
            analyse_text = "L'information soumise est exacte et validée par le consensus scientifique actuel. Les recherches climatiques corroborent formellement cette dynamique. Ces observations soulignent la nécessité d'intégrer ces risques dans les plans d'adaptation locaux et les politiques de résilience."
        elif verdict == "REFUTE":
            badge_class = "badge-refuted"
            badge_icon = "❌"
            badge_text = "RÉFUTÉ / DÉSINFORMATION"
            analyse_text = "L'information soumise est inexacte ou trompeuse. Les données climatologiques démentent formellement cette déclaration. Il est crucial de corriger cette communication afin de ne pas fausser l'évaluation des vulnérabilités climatiques."
        else:
            badge_class = "badge-insufficient"
            badge_icon = "⚠️"
            if similarity_score >= 0.20:
                badge_text = "PREUVES INDIRECTES / INSUFFISANTES"
                analyse_text = "Les documents institutionnels (GIEC, OMM, etc.) traitent de sujets connexes, mais ils ne permettent pas de confirmer ou de réfuter explicitement et directement cette affirmation précise. Une analyse humaine des documents sourcés ci-dessous est recommandée."
            else:
                badge_text = "AUCUNE PREUVE SCIENTIFIQUE"
                analyse_text = "Aucune source institutionnelle ne mentionne ou ne justifie cette affirmation. En l'absence de données fiables et directes issues de la littérature scientifique officielle (GIEC, OMM, rapports nationaux), cette déclaration est considérée comme totalement infondée."

        # Préparation des sources
        sources = []
        if similarity_score >= 0.20:
            # Sélection des sources à AFFICHER (top-k). Par défaut, identique à
            # avant : les k meilleurs candidats du retrieval sémantique déjà
            # effectué ci-dessus (indices/distances), sans aucun changement.
            #
            # Si zone_geo contient au moins une entité géographique reconnue
            # (même lexique que relevance_filter.py), on repondère un pool de
            # candidats plus large en ajoutant ZONE_GEO_BOOST au score cosinus
            # des candidats dont l'evidence mentionne cette zone, puis on
            # retrie et reprend les k meilleurs. C'est un second passage
            # strictement APRÈS le choix du top-1 utilisé pour le seuil
            # anti-hallucination et la classification (inchangés ci-dessus) :
            # zone_geo ne peut donc jamais faire basculer le verdict, il
            # affecte uniquement quelles sources complémentaires sont mises
            # en avant. Si zone_geo n'est pas reconnu (valeur par défaut,
            # faute de frappe, zone hors lexique), aucune repondération n'a
            # lieu et le comportement reste strictement identique à avant.
            source_indices = [int(i) for i in indices[0][:k]]
            zone_entities = extract_entities(request.zone_geo)
            if zone_entities:
                pool_size = len(corpus_df)  # corpus institutionnel réduit (~4870 lignes) : un re-scan complet reste négligeable en coût
                pool_distances, pool_indices = index.search(c_emb, pool_size)
                candidates = []
                for idx, score in zip(pool_indices[0], pool_distances[0]):
                    idx = int(idx)
                    evidence_text = str(corpus_df.iloc[idx]['evidence'])
                    boosted_score = float(score)
                    if zone_entities & extract_entities(evidence_text):
                        boosted_score += ZONE_GEO_BOOST
                    candidates.append((boosted_score, idx))
                candidates.sort(key=lambda pair: pair[0], reverse=True)
                source_indices = [idx for _, idx in candidates[:k]]

            for idx in source_indices:
                row = corpus_df.iloc[idx]
                evidence_text = str(row['evidence'])
                sources.append(Source(
                    institution=str(row.get('institution', 'Source Inconnue')),
                    evidence=evidence_text,
                    title=str(row.get('title', 'Document officiel')),
                    year=str(row.get('year', 'N/A')),
                    url=str(row.get('url', '#')),
                    # Filtre de cohérence géo/thématique appliqué uniquement à
                    # l'affichage, une fois le verdict déjà déterminé.
                    relevance_uncertain=is_relevance_uncertain(request.claim, evidence_text)
                ))

        return VerificationResponse(
            badge_class=badge_class,
            badge_icon=badge_icon,
            badge_text=badge_text,
            analyse_text=analyse_text,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # La casse de l'extension n'est pas fiable comme signal de format : de
    # nombreux PDF (export Windows/macOS, scanners) sont nommés ".PDF" ou
    # ".Pdf". Un filtre sensible à la casse rejetait ces fichiers, pourtant
    # valides, avec un message trompeur ("format non accepté").
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    is_txt = filename_lower.endswith('.txt')
    if not is_pdf and not is_txt:
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF et TXT sont acceptés.")

    try:
        content = await file.read()
        text = ""

        if is_pdf:
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            except Exception:
                # Erreur de format côté client (PDF corrompu/invalide), pas une
                # panne serveur : 400 plutôt que 500, avec un message actionnable.
                raise HTTPException(
                    status_code=400,
                    detail="Le fichier PDF est illisible ou corrompu. Vérifiez qu'il s'agit bien d'un PDF valide et réessayez."
                )

            # Parcourt TOUTES les pages (auparavant limité aux 5 premières, ce
            # qui tronquait silencieusement tout document plus long). Une page
            # individuellement illisible par PyPDF2 (scan/image sans couche
            # texte, page corrompue) déclenche un fallback OCR SUR CETTE SEULE
            # PAGE avant d'être comptée comme vide — jamais systématique : le
            # rendu image + OCR est coûteux en temps, donc réservé aux pages
            # où l'extraction native a échoué.
            page_texts = []
            failed_pages = 0
            ocr_pages = 0
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                except Exception:
                    page_text = None

                if not page_text and OCR_AVAILABLE:
                    try:
                        images = convert_from_bytes(content, first_page=page_num, last_page=page_num, dpi=200)
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0], lang="fra+eng")
                            if ocr_text and ocr_text.strip():
                                page_text = ocr_text
                                ocr_pages += 1
                    except Exception as ocr_err:
                        print(f"[upload-pdf] OCR échoué sur la page {page_num} de {file.filename}: {ocr_err}")

                if page_text:
                    page_texts.append(page_text)
                else:
                    failed_pages += 1
            text = " ".join(page_texts)
            print(f"[upload-pdf] {file.filename}: {len(pdf_reader.pages)} page(s), "
                  f"{failed_pages} page(s) sans texte extrait, {ocr_pages} page(s) récupérée(s) par OCR, "
                  f"{len(text)} caractères extraits au total")

            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Aucun texte n'a pu être extrait de ce PDF (pages scannées/images sans OCR ?)."
                )
        else:
            text = content.decode('utf-8')

        # Le champ extracted_text préremplit la barre de claim (pas un
        # visualiseur de document) : on renvoie un aperçu court plutôt que le
        # texte intégral, mais en échantillonnant début ET fin du document
        # extrait (pas seulement les premiers caractères) pour rester
        # représentatif d'un document long.
        HEAD, TAIL = 350, 150
        if len(text) <= HEAD + TAIL:
            extracted = text
        else:
            extracted = text[:HEAD].rstrip() + " [...] " + text[-TAIL:].lstrip()

        return {"extracted_text": extracted.strip()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de lecture du document: {str(e)}")
