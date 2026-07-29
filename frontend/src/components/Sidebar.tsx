import React, { useRef, useState } from 'react';
import { Globe, UploadCloud, CheckCircle2 } from 'lucide-react';

interface SidebarProps {
  zoneGeo: string;
  setZoneGeo: (zone: string) => void;
  onPdfUpload: (file: File) => void;
}

export default function Sidebar({ zoneGeo, setZoneGeo, onPdfUpload }: SidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onPdfUpload(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onPdfUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <aside className="w-80 bg-white border-r border-[#E2E8F0] p-6 flex flex-col h-full shadow-sm">
      {/* Brand Header */}
      <div className="flex items-center gap-3 mb-12">
        <Globe className="text-[#059669] w-10 h-10" />
        <div>
          <h1 className="text-xl font-extrabold text-[#0F172A] leading-tight">TERRAVA-AI <span className="text-[#059669]">Pro</span></h1>
          <p className="text-xs text-[#64748B] font-medium">Fact-Checking Climatique</p>
        </div>
      </div>

      {/* Filtre Régional */}
      <div className="mb-10">
        <label className="block text-sm font-bold text-[#1E293B] uppercase tracking-wide mb-3">📍 Zone Géographique</label>
        <select 
          value={zoneGeo} 
          onChange={(e) => setZoneGeo(e.target.value)}
          className="w-full bg-[#F8FAFC] border border-[#CBD5E1] rounded-lg p-3 text-sm focus:outline-none focus:ring-2 focus:ring-[#059669]/20 focus:border-[#059669] transition-all cursor-pointer"
        >
          <option>Global (International)</option>
          <option>Afrique de l'Ouest</option>
          <option>Côte d'Ivoire</option>
        </select>
      </div>

      {/* Analyse PDF */}
      <div>
        <label className="block text-sm font-bold text-[#1E293B] uppercase tracking-wide mb-3">📄 Analyse de Document</label>
        <p className="text-xs text-[#64748B] mb-4">Importez un PDF pour extraire une affirmation à vérifier.</p>
        
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all group ${
            isDragging
              ? "bg-[#F0FDF4] border-[#059669]"
              : "border-[#CBD5E1] hover:bg-[#F8FAFC] hover:border-[#059669]"
          }`}
        >
          <UploadCloud className="w-8 h-8 text-[#94A3B8] mb-2 group-hover:text-[#059669] transition-colors" />
          <span className="text-sm font-medium text-[#475569]">Glissez un PDF ici ou</span>
          <span className="text-sm font-bold text-[#059669] mt-1">Parcourez vos fichiers</span>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange} 
            accept=".pdf,.txt" 
            className="hidden" 
          />
        </div>
      </div>
      
      <div className="mt-auto">
        <div className="bg-[#F0FDF4] border border-[#BBF7D0] rounded-lg p-4 flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-[#16A34A] flex-shrink-0 mt-0.5" />
          <p className="text-xs text-[#166534] font-medium leading-relaxed">
            Connecté au moteur d'intelligence climatique. Données synchronisées.
          </p>
        </div>
      </div>
    </aside>
  );
}
