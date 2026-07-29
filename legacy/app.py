import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import joblib
import PyPDF2
import textwrap

# --- 1. CONFIGURATION DE L'INTERFACE ---
st.set_page_config(
    page_title="ClimaCheck Pro - Fact-Checking Climatique", 
    page_icon="🌍", 
    layout="wide"
)

# --- 2. CSS SUR-MESURE & ASSETS (HTML5 / CSS3 Propre) ---
SVG_LOGO = """
<svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 12H22" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M12 2C14.5013 4.73835 15.9228 8.29203 16 12C15.9228 15.708 14.5013 19.2616 12 22C9.49872 19.2616 8.07725 15.708 8 12C8.07725 8.29203 9.49872 4.73835 12 2Z" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M12 12C12 12 14.5 9 17 9C19.5 9 20 12 20 12C20 12 17.5 15 15 15C12.5 15 12 12 12 12Z" fill="#10B981" opacity="0.8"/>
</svg>
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Typographie et fond global */
.stApp {
    background-color: #F8FAFC;
    font-family: 'Inter', sans-serif;
    color: #0F172A;
}

/* Bouton principal stylisé */
div.stButton > button:first-child {
    background-color: #059669 !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
}
div.stButton > button:first-child:hover {
    background-color: #047857 !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
    transform: translateY(-1px);
}

/* Zone de texte stylisée */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid #CBD5E1 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    font-size: 1.05rem !important;
    padding: 16px !important;
}
.stTextArea textarea:focus {
    border-color: #059669 !important;
    box-shadow: 0 0 0 2px rgba(5, 150, 105, 0.2) !important;
}

/* --- COMPOSANTS HTML --- */
.clima-card {
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
    padding: 32px;
    margin-top: 16px;
    margin-bottom: 32px;
    border: 1px solid #E2E8F0;
    color: #0F172A;
}
.verdict-header {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin-bottom: 24px;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 20px;
}
.verdict-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 16px;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge {
    padding: 10px 24px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: white;
}
.badge-confirmed { background-color: #059669; box-shadow: 0 4px 10px rgba(5,150,105,0.2); }
.badge-refuted { background-color: #DC2626; box-shadow: 0 4px 10px rgba(220,38,38,0.2); }
.badge-insufficient { background-color: #D97706; box-shadow: 0 4px 10px rgba(217,119,6,0.2); }

.analysis-content {
    font-size: 1.1rem;
    line-height: 1.7;
    margin-bottom: 32px;
    color: #334155;
}
.analysis-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 12px;
}
.sources-container {
    background-color: #F8FAFC;
    border-radius: 8px;
    padding: 24px;
    border: 1px solid #E2E8F0;
}
.sources-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.source-item {
    margin-bottom: 16px;
    padding: 16px;
    background: white;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #059669;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.source-item-insufficient {
    border-left: 4px solid #D97706;
}
.source-label {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 700;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.source-text {
    font-size: 0.95rem;
    color: #334155;
    font-style: italic;
    line-height: 1.6;
}
.source-meta {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 12px;
    display: flex;
    justify-content: flex-end;
}
.source-meta a {
    color: #059669;
    text-decoration: none;
    font-weight: 600;
}
.source-meta a:hover {
    text-decoration: underline;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 3. CHARGEMENT SÉCURISÉ DES DONNÉES ---
@st.cache_resource(show_spinner=False)
def load_verification_engine():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("models_saved/faiss_index.bin")
    classifier = joblib.load("models_saved/classifier.joblib")
    corpus_df = pd.read_csv("data/corpus.csv")
    return embedding_model, index, classifier, corpus_df

def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "".join(page.extract_text() for page in pdf_reader.pages)
    return uploaded_file.getvalue().decode("utf-8")

try:
    embedding_model, index, classifier, corpus_df = load_verification_engine()
except Exception:
    st.error("Le système documentaire est actuellement indisponible.")
    st.stop()


# --- 4. BARRE LATÉRALE PRO (SIDEBAR) ---
with st.sidebar:
    sidebar_header = f"""
<div style='display:flex; align-items:center; gap:10px; margin-bottom: 20px;'>
{SVG_LOGO}
<span style='font-size:1.5rem; font-weight:800; color:#0F172A;'>ClimaCheck <span style='color:#059669;'>Pro</span></span>
</div>
"""
    st.markdown(sidebar_header, unsafe_allow_html=True)
    
    st.markdown("### 📍 Zone Géographique Prioritaire")
    zone_geo = st.selectbox(
        "Filtrer l'analyse :", 
        ["Global (International)", "Afrique de l'Ouest", "Côte d'Ivoire"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📚 Traçabilité du Dataset")
    if 'institution' in corpus_df.columns:
        df_sources = corpus_df[['institution', 'url']].drop_duplicates().dropna()
        sources_list_html = "<ul style='padding-left: 20px; font-size: 0.9rem; color: #475569;'>"
        for _, row in df_sources.iterrows():
            url = row['url'] if str(row['url']).startswith("http") else "#"
            sources_list_html += f"<li style='margin-bottom: 8px;'><a href='{url}' target='_blank' style='color: #059669; text-decoration: none; font-weight: 600;'>{row['institution']}</a></li>"
        sources_list_html += "</ul>"
        st.markdown(sources_list_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📄 Analyse de Document (PDF)")
    st.write("Importez un document pour en extraire une affirmation à vérifier.")
    uploaded_file = st.file_uploader("Fichier PDF ou TXT", type=["txt", "pdf"], label_visibility="collapsed")

    claim_text = ""
    if uploaded_file:
        raw_text = extract_text_from_file(uploaded_file)
        claim_text = raw_text[:500] + ("..." if len(raw_text) > 500 else "")


# --- 5. EN-TÊTE PRINCIPAL (HEADER HTML) ---
header_html = f"""
<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 32px; padding: 20px; background: white; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
{SVG_LOGO}
<div>
<h1 style="margin: 0; color: #0F172A; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">ClimaCheck <span style="color: #059669;">Pro</span></h1>
<p style="margin: 4px 0 0 0; color: #64748B; font-size: 1.1rem; font-weight: 500;">Plateforme d'Intelligence et de Fact-Checking Climatique</p>
</div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Zone de saisie (Formulaire natif avec CSS personnalisé)
with st.form("verification_form"):
    claim = st.text_input(
        "Déclaration à vérifier scientifiquement (Appuyez sur Entrée pour lancer) :", 
        value=claim_text, 
        placeholder="Ex: Les précipitations extrêmes vont s'intensifier en Afrique de l'Ouest d'ici 2050..."
    )
    submitted = st.form_submit_button("Vérifier la déclaration")


# --- 6. LOGIQUE DE VÉRIFICATION & ZÉRO JARGON ---
if submitted:
    if not claim.strip():
        st.warning("Veuillez saisir une déclaration ou importer un document pour lancer l'analyse.")
    else:
        with st.status("Recherche dans les rapports institutionnels en cours...", expanded=True) as status:
            st.write(f"🔍 Filtrage des données pour : **{zone_geo}**...")
            
            c_emb = embedding_model.encode([claim], normalize_embeddings=True)
            k = 3
            distances, indices = index.search(c_emb, k)
            
            top_evidence_row = corpus_df.iloc[indices[0][0]]
            top_evidence = top_evidence_row['evidence']
            similarity_score = distances[0][0]
            
            st.write("⚖️ Analyse d'impact climatique...")
            
            # GARDE-FOU ANTI-HALLUCINATION : Filtrage de pertinence (Tolérance augmentée)
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
                    
            status.update(label="Analyse scientifique terminée.", state="complete", expanded=False)

        # --- 7. GÉNÉRATION DE LA FICHE DE SYNTHÈSE HTML ---
        if verdict == "CONFIRME":
            badge_class = "badge-confirmed"
            badge_icon = "✅"
            badge_text = "CONFIRMÉ PAR LES DONNÉES SCIENTIFIQUES"
            analyse_text = f"L'information soumise est exacte et validée par le consensus scientifique actuel. Les recherches climatiques corroborent formellement cette dynamique. Ces observations soulignent la nécessité d'intégrer ces risques dans les plans d'adaptation locaux et les politiques de résilience."
        
        elif verdict == "REFUTE":
            badge_class = "badge-refuted"
            badge_icon = "❌"
            badge_text = "RÉFUTÉ / DÉSINFORMATION"
            analyse_text = f"L'information soumise est inexacte ou trompeuse. Les données climatologiques démentent formellement cette déclaration. Il est crucial de corriger cette communication afin de ne pas fausser l'évaluation des vulnérabilités climatiques."
        
        else:
            badge_class = "badge-insufficient"
            badge_icon = "⚠️"
            if similarity_score >= 0.20:
                badge_text = "PREUVES INDIRECTES / INSUFFISANTES"
                analyse_text = "Les documents institutionnels (GIEC, OMM, etc.) traitent de sujets connexes, mais ils ne permettent pas de confirmer ou de réfuter explicitement et directement cette affirmation précise. Une analyse humaine des documents sourcés ci-dessous est recommandée."
            else:
                badge_text = "AUCUNE PREUVE SCIENTIFIQUE"
                analyse_text = "Aucune source institutionnelle ne mentionne ou ne justifie cette affirmation. En l'absence de données fiables et directes issues de la littérature scientifique officielle (GIEC, OMM, rapports nationaux), cette déclaration est considérée comme totalement infondée."

        # Rendu des sources
        sources_html = ""
        if similarity_score >= 0.20:
            for i in range(k):
                row = corpus_df.iloc[indices[0][i]]
                ev = row['evidence']
                inst = row.get('institution', 'Source Inconnue')
                title = row.get('title', 'Document officiel')
                url = row.get('url', '#')
                url_link = f"<a href='{url}' target='_blank'>Consulter l'archive ↗</a>" if str(url).startswith("http") else ""
                
                sources_html += f"""
<div class="source-item">
<div class="source-label">{inst}</div>
<div class="source-text">« {ev} »</div>
<div class="source-meta">{title} | {url_link}</div>
</div>"""
        else:
            sources_html = """
<div class="source-item source-item-insufficient">
<div class="source-text">
Le moteur d'intelligence climatique n'a trouvé aucun rapprochement avec une donnée institutionnelle validée. L'information n'a aucune traçabilité scientifique avérée.
</div>
</div>"""

        final_html = f"""
<article class="clima-card">
<header class="verdict-header">
<div class="verdict-title">Fiche de Synthèse</div>
<div class="badge {badge_class}">
<span>{badge_icon}</span>
<span>{badge_text}</span>
</div>
</header>
<section class="analysis-content">
<div class="analysis-title">Synthèse des Faits & Impacts :</div>
<div>{analyse_text}</div>
</section>
<section class="sources-container">
<div class="sources-title">
<svg width="22" height="22" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path></svg>
Traçabilité Officielle
</div>
{sources_html}
</section>
</article>
"""
        st.markdown(final_html, unsafe_allow_html=True)
