import React from 'react';
import { Search, Loader2 } from 'lucide-react';

interface ClaimInputProps {
  claim: string;
  setClaim: (val: string) => void;
  onVerify: (text: string) => void;
  isLoading: boolean;
}

export default function ClaimInput({ claim, setClaim, onVerify, isLoading }: ClaimInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onVerify(claim);
    }
  };

  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-[#E2E8F0] relative focus-within:ring-2 focus-within:ring-[#059669]/20 focus-within:border-[#059669] transition-all">
      <textarea
        value={claim}
        onChange={(e) => setClaim(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Collez ou saisissez la déclaration scientifique à vérifier... (Appuyez sur Entrée)"
        className="w-full resize-none outline-none text-lg text-[#334155] placeholder:text-[#94A3B8] min-h-[120px] bg-transparent"
      />
      <div className="absolute bottom-6 right-6">
        <button 
          onClick={() => onVerify(claim)}
          disabled={isLoading || !claim.trim()}
          className="flex items-center gap-2 bg-[#059669] hover:bg-[#047857] disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-xl shadow-md hover:shadow-lg transition-all"
        >
          {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
          Vérifier la déclaration
        </button>
      </div>
    </div>
  );
}
