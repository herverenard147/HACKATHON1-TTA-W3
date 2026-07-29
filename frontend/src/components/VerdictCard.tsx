import React, { useState } from 'react';
import { Share2, Check, ThumbsUp, ThumbsDown } from 'lucide-react';
import { shareOrCopy } from '../shareText';
import { API_BASE_URL } from '../config';

export default function VerdictCard({ result, userId }: { result: any; userId: string }) {
  const [copied, setCopied] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState<'up' | 'down' | null>(null);

  const handleFeedback = async (rating: 'up' | 'down') => {
    if (!result.verification_id || feedbackSent) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verification_id: result.verification_id, user_id: userId, rating }),
      });
      if (res.ok) setFeedbackSent(rating);
    } catch (err) {
      // Le feedback est une fonctionnalité annexe : un échec silencieux ici
      // ne doit pas interrompre la lecture du verdict déjà affiché.
    }
  };

  // Styles dynamiques en fonction du verdict
  let badgeColor = "bg-[#D97706] text-white"; // Ambre par défaut

  if (result.badge_class === "badge-confirmed") badgeColor = "bg-[#059669] text-white";
  else if (result.badge_class === "badge-refuted") badgeColor = "bg-[#DC2626] text-white";

  const handleShare = async () => {
    const usedClipboard = await shareOrCopy(result);
    if (usedClipboard) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
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

      {/* Feedback 👍/👎 : uniquement actionnable si la vérification a été
          sauvegardée (verification_id présent, i.e. un user_id a été fourni,
          voir main.py). Simplement collecté pour l'instant (voir
          DOCUMENTATION_TECHNIQUE.md), pas encore exploité. */}
      {result.verification_id && (
        <div className="mt-6 pt-6 border-t border-[#F1F5F9] flex items-center gap-3">
          <span className="text-sm text-[#64748B] font-medium">Ce verdict vous a-t-il été utile ?</span>
          <button
            onClick={() => handleFeedback('up')}
            disabled={!!feedbackSent}
            className={`p-2 rounded-lg transition-all ${feedbackSent === 'up' ? 'bg-[#059669] text-white' : 'bg-[#F1F5F9] text-[#64748B] hover:bg-[#E2E8F0]'} disabled:cursor-not-allowed`}
          >
            <ThumbsUp className="w-4 h-4" />
          </button>
          <button
            onClick={() => handleFeedback('down')}
            disabled={!!feedbackSent}
            className={`p-2 rounded-lg transition-all ${feedbackSent === 'down' ? 'bg-[#DC2626] text-white' : 'bg-[#F1F5F9] text-[#64748B] hover:bg-[#E2E8F0]'} disabled:cursor-not-allowed`}
          >
            <ThumbsDown className="w-4 h-4" />
          </button>
          {feedbackSent && <span className="text-xs text-[#94A3B8]">Merci pour votre retour !</span>}
        </div>
      )}
    </div>
  );
}
