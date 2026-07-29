import React, { useState } from 'react';
import { Share2, Check } from 'lucide-react';

// Texte formaté, pensé pour être collé tel quel sur WhatsApp/Facebook/X : pas
// de mise en forme HTML (ces canaux ne la rendent pas), juste des sauts de
// ligne et des puces simples. Reprend le claim d'origine, le verdict et
// jusqu'à 2 sources principales (cf. consigne "1-2 sources principales") —
// jamais tronqué au milieu d'une phrase : seul l'extrait de chaque evidence
// est raccourci (à une limite de mots, pas de caractères bruts) si trop long.
function buildShareText(result: any): string {
  const lines: string[] = [];
  lines.push('🌍 TERRAVA-AI — Vérification climatique');
  lines.push('');
  if (result.claim) {
    lines.push('Affirmation vérifiée :');
    lines.push(`« ${result.claim} »`);
    lines.push('');
  }
  lines.push(`Verdict : ${result.badge_icon} ${result.badge_text}`);
  lines.push('');
  lines.push(result.analyse_text);

  const sources = (result.sources || []).slice(0, 2);
  if (sources.length > 0) {
    lines.push('');
    lines.push('Sources :');
    sources.forEach((s: any, i: number) => {
      let excerpt = s.evidence;
      const LIMIT = 180;
      if (excerpt.length > LIMIT) {
        const cut = excerpt.slice(0, LIMIT);
        excerpt = cut.slice(0, cut.lastIndexOf(' ')) + '…';
      }
      lines.push(`${i + 1}. ${s.institution} — « ${excerpt} »`);
    });
  }

  lines.push('');
  lines.push('Vérifié avec TERRAVA-AI (fact-checking climatique)');
  return lines.join('\n');
}

export default function VerdictCard({ result }: { result: any }) {
  const [copied, setCopied] = useState(false);

  // Styles dynamiques en fonction du verdict
  let badgeColor = "bg-[#D97706] text-white"; // Ambre par défaut

  if (result.badge_class === "badge-confirmed") badgeColor = "bg-[#059669] text-white";
  else if (result.badge_class === "badge-refuted") badgeColor = "bg-[#DC2626] text-white";

  const handleShare = async () => {
    const text = buildShareText(result);

    // API de partage native (WhatsApp/X/etc. apparaissent directement dans le
    // menu de partage sur mobile) quand disponible ; sinon presse-papier,
    // qui fonctionne partout (desktop inclus) sans dépendance supplémentaire.
    if (navigator.share) {
      try {
        await navigator.share({ text });
        return;
      } catch (err) {
        // Partage annulé par l'utilisateur ou API refusée dans ce contexte :
        // on retombe silencieusement sur le presse-papier ci-dessous.
      }
    }

    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      alert("Impossible de copier automatiquement. Voici le texte à partager :\n\n" + text);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-[#E2E8F0] p-8 mb-8">
      <div className="flex items-start justify-between border-b border-[#F1F5F9] pb-6 mb-6">
        <div className="flex flex-col items-start">
          <h2 className="text-sm font-bold text-[#64748B] uppercase tracking-widest mb-4">Synthèse de Fact-Checking</h2>
          <div className={`px-6 py-2.5 rounded-full font-bold text-sm tracking-wide shadow-sm flex items-center gap-2 ${badgeColor}`}>
            <span>{result.badge_icon}</span>
            {result.badge_text}
          </div>
        </div>
        <button
          onClick={handleShare}
          className="flex items-center gap-2 bg-[#F1F5F9] hover:bg-[#E2E8F0] text-[#334155] font-semibold py-2.5 px-4 rounded-xl transition-all shrink-0"
        >
          {copied ? <Check className="w-4 h-4 text-[#059669]" /> : <Share2 className="w-4 h-4" />}
          {copied ? 'Copié !' : 'Partager'}
        </button>
      </div>

      <div>
        <h3 className="text-xl font-bold text-[#0F172A] mb-3">Synthèse des Faits & Impacts :</h3>
        <p className="text-[#334155] text-lg leading-relaxed">
          {result.analyse_text}
        </p>
      </div>

      {/* Détails techniques : uniquement présents pour les niveaux
          amateur/expert (technical_details est null pour débutant/intermédiaire,
          voir main.py build_analyse_text). */}
      {result.technical_details && (
        <div className="mt-6 pt-6 border-t border-[#F1F5F9] text-sm text-[#64748B] font-mono space-y-1">
          <div>Score de similarité (cosinus) : {result.technical_details.similarity_score}</div>
          <div>Sources consultées : {result.technical_details.nb_sources_consulted}</div>
          {result.technical_details.raw_nli_class && (
            <div>Classe NLI : {result.technical_details.raw_nli_class}</div>
          )}
          {result.technical_details.class_probabilities && (
            <div>
              Probabilités : {Object.entries(result.technical_details.class_probabilities)
                .map(([cls, p]) => `${cls}=${p}`)
                .join(', ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
