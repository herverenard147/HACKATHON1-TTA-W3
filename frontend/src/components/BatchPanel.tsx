import React, { useState } from 'react';
import { Layers, X, Loader2 } from 'lucide-react';
import { API_BASE_URL } from '../config';

interface BatchResultItem {
  claim: string;
  result: any;
}

// Vérification par lot (phase 5) : un texte collé, une affirmation par
// ligne (découpage naïf, pas de segmentation NLP par phrase - voir
// check_claims_batch dans main.py pour les limites assumées de ce
// découpage). Chaque ligne suit le pipeline complet côté backend.
export default function BatchPanel({
  zoneGeo,
  comprehensionLevel,
  userId,
  onClose,
}: {
  zoneGeo: string;
  comprehensionLevel: string;
  userId: string;
  onClose: () => void;
}) {
  const [text, setText] = useState('');
  const [results, setResults] = useState<BatchResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    setResults(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/check-claims-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, zone_geo: zoneGeo, comprehension_level: comprehensionLevel, user_id: userId }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erreur lors de la vérification par lot.');
      setResults(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-2xl w-full max-h-[85vh] overflow-y-auto p-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-[#1E293B] flex items-center gap-2">
            <Layers className="w-5 h-5" /> Vérification par lot
          </h2>
          <button onClick={onClose} className="text-[#94A3B8] hover:text-[#334155]">
            <X className="w-5 h-5" />
          </button>
        </div>
        <p className="text-sm text-[#64748B] mb-4">
          Collez plusieurs affirmations, une par ligne. Chacune suit le pipeline complet de vérification (jusqu'à 20 par lot).
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={"Affirmation 1...\nAffirmation 2...\nAffirmation 3..."}
          className="w-full min-h-[140px] border border-[#CBD5E1] rounded-lg p-3 text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-[#059669]/20 focus:border-[#059669]"
        />

        <button
          onClick={handleSubmit}
          disabled={loading || !text.trim()}
          className="flex items-center gap-2 bg-[#059669] hover:bg-[#047857] disabled:opacity-50 text-white font-semibold py-2.5 px-5 rounded-xl transition-all mb-6"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Layers className="w-4 h-4" />}
          Vérifier le lot
        </button>

        {error && <p className="text-red-600 text-sm mb-4">{error}</p>}

        {results && (
          <div className="space-y-3">
            {results.map((item, idx) => (
              <div key={idx} className="border border-[#E2E8F0] rounded-xl p-4">
                <p className="text-xs font-bold text-[#64748B] uppercase mb-1">
                  {item.result.badge_icon} {item.result.badge_text}
                </p>
                <p className="text-sm text-[#334155]">« {item.claim} »</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
