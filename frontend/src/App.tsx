import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import ClaimInput from './components/ClaimInput';
import VerdictCard from './components/VerdictCard';
import SourcesAccordion from './components/SourcesAccordion';
import { API_BASE_URL } from './config';

export default function App() {
  const [claim, setClaim] = useState("");
  const [zoneGeo, setZoneGeo] = useState("Global (International)");
  const [comprehensionLevel, setComprehensionLevel] = useState("intermédiaire");
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const handleVerify = async (text: string) => {
    if (!text.trim()) return;
    setIsVerifying(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/check-claim`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: text, zone_geo: zoneGeo, comprehension_level: comprehensionLevel })
      });
      if (!response.ok) throw new Error("Erreur serveur lors de la vérification.");
      const data = await response.json();
      // On rattache la déclaration vérifiée à `result` (l'API ne la renvoie pas
      // elle-même) : nécessaire pour que l'export/partage du verdict puisse
      // citer l'affirmation d'origine sans dépendre du state `claim`, qui peut
      // changer entre-temps si l'utilisateur retape quelque chose.
      setResult({ ...data, claim: text });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsVerifying(false);
    }
  };

  const handlePdfUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/upload-pdf`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Erreur lors de la lecture du document.");
      setClaim(data.extracted_text);
    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div className="flex h-screen bg-[#F8FAFC] text-[#0F172A] font-inter">
      <Sidebar
        zoneGeo={zoneGeo}
        setZoneGeo={setZoneGeo}
        comprehensionLevel={comprehensionLevel}
        setComprehensionLevel={setComprehensionLevel}
        onPdfUpload={handlePdfUpload}
      />
      
      <main className="flex-1 overflow-y-auto p-8 lg:p-12">
        <div className="max-w-4xl mx-auto space-y-8">
          
          <header className="mb-12">
            <h1 className="text-3xl font-extrabold tracking-tight mb-2">Analyse Climatique</h1>
            <p className="text-[#64748B] text-lg">Vérifiez la validité scientifique d'une déclaration par rapport aux données du GIEC et de l'OMM.</p>
          </header>

          <ClaimInput claim={claim} setClaim={setClaim} onVerify={handleVerify} isLoading={isVerifying} />
          
          {error && (
            <div className="bg-red-50 text-red-700 p-4 rounded-xl border border-red-200">
              {error}
            </div>
          )}

          {result && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <VerdictCard result={result} />
              <SourcesAccordion sources={result.sources} />
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
