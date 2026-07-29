import React from 'react';
import { BookOpen, ExternalLink, AlertTriangle } from 'lucide-react';
import { API_BASE_URL } from '../config';

// Construit une URL "consultable" à partir de la valeur url renvoyée par l'API :
// - une URL absolue externe (http/https) est utilisée telle quelle.
// - un chemin relatif (ex. "/documents/xxx.txt") est un document servi par le
//   backend (voir le mount StaticFiles dans main.py) et doit être préfixé par
//   l'URL du backend, pas résolu contre l'origine du frontend.
// - toute autre valeur ("#", vide, placeholder) n'est pas un lien exploitable.
function resolveArchiveUrl(url: string | undefined): string | null {
  if (!url || url === '#') return null;
  if (url.startsWith('http://') || url.startsWith('https://')) return url;
  if (url.startsWith('/')) return `${API_BASE_URL}${url}`;
  return null;
}

export default function SourcesAccordion({ sources }: { sources: any[] }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="bg-[#F8FAFC] rounded-2xl border border-[#E2E8F0] p-8">
      <div className="flex items-center gap-3 mb-6">
        <BookOpen className="w-6 h-6 text-[#475569]" />
        <h2 className="text-lg font-bold text-[#1E293B] uppercase tracking-wide">Traçabilité Officielle</h2>
      </div>
      
      <div className="space-y-4">
        {sources.map((src, idx) => {
          const archiveUrl = resolveArchiveUrl(src.url);
          return (
            <div key={idx} className={`bg-white p-5 rounded-xl shadow-sm border border-[#E2E8F0] border-l-4 ${src.relevance_uncertain ? 'border-l-[#D97706]' : 'border-l-[#059669]'}`}>
              <div className="text-xs font-bold text-[#64748B] uppercase mb-2 flex items-center gap-2">
                {src.institution} — {src.year !== 'nan' ? src.year : 'Document officiel'}
              </div>
              {src.relevance_uncertain && (
                <div className="flex items-center gap-1.5 text-xs font-semibold text-[#B45309] bg-[#FFFBEB] border border-[#FDE68A] rounded-lg px-3 py-1.5 mb-3 w-fit">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  Pertinence géographique/thématique incertaine
                </div>
              )}
              <p className="text-[#334155] italic leading-relaxed mb-4">
                « {src.evidence} »
              </p>
              <div className="flex justify-between items-center text-sm border-t border-[#F1F5F9] pt-3 mt-3">
                <span className="text-[#94A3B8] font-medium">{src.title}</span>
                {archiveUrl ? (
                  <a href={archiveUrl} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-[#059669] font-bold hover:underline">
                    Consulter l'archive <ExternalLink className="w-4 h-4" />
                  </a>
                ) : (
                  <span className="text-[#94A3B8] italic">Document source interne — non disponible en ligne</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
