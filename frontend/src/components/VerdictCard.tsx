import React from 'react';

export default function VerdictCard({ result }: { result: any }) {
  // Styles dynamiques en fonction du verdict
  let badgeColor = "bg-[#D97706] text-white"; // Ambre par défaut
  
  if (result.badge_class === "badge-confirmed") badgeColor = "bg-[#059669] text-white";
  else if (result.badge_class === "badge-refuted") badgeColor = "bg-[#DC2626] text-white";

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-[#E2E8F0] p-8 mb-8">
      <div className="flex flex-col items-start border-b border-[#F1F5F9] pb-6 mb-6">
        <h2 className="text-sm font-bold text-[#64748B] uppercase tracking-widest mb-4">Synthèse de Fact-Checking</h2>
        <div className={`px-6 py-2.5 rounded-full font-bold text-sm tracking-wide shadow-sm flex items-center gap-2 ${badgeColor}`}>
          <span>{result.badge_icon}</span>
          {result.badge_text}
        </div>
      </div>
      
      <div>
        <h3 className="text-xl font-bold text-[#0F172A] mb-3">Synthèse des Faits & Impacts :</h3>
        <p className="text-[#334155] text-lg leading-relaxed">
          {result.analyse_text}
        </p>
      </div>
    </div>
  );
}
