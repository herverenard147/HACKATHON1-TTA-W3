import React from 'react';
import { BookOpen, ExternalLink } from 'lucide-react';

export default function SourcesAccordion({ sources }: { sources: any[] }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="bg-[#F8FAFC] rounded-2xl border border-[#E2E8F0] p-8">
      <div className="flex items-center gap-3 mb-6">
        <BookOpen className="w-6 h-6 text-[#475569]" />
        <h2 className="text-lg font-bold text-[#1E293B] uppercase tracking-wide">Traçabilité Officielle</h2>
      </div>
      
      <div className="space-y-4">
        {sources.map((src, idx) => (
          <div key={idx} className="bg-white p-5 rounded-xl shadow-sm border border-[#E2E8F0] border-l-4 border-l-[#059669]">
            <div className="text-xs font-bold text-[#64748B] uppercase mb-2">
              {src.institution} — {src.year !== 'nan' ? src.year : 'Document officiel'}
            </div>
            <p className="text-[#334155] italic leading-relaxed mb-4">
              « {src.evidence} »
            </p>
            <div className="flex justify-between items-center text-sm border-t border-[#F1F5F9] pt-3 mt-3">
              <span className="text-[#94A3B8] font-medium">{src.title}</span>
              {src.url !== "#" && (
                <a href={src.url} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-[#059669] font-bold hover:underline">
                  Consulter l'archive <ExternalLink className="w-4 h-4" />
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
