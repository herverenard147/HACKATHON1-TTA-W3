// Texte formaté, pensé pour être collé tel quel sur WhatsApp/Facebook/X : pas
// de mise en forme HTML (ces canaux ne la rendent pas), juste des sauts de
// ligne et des puces simples. Reprend le claim d'origine, le verdict et
// jusqu'à 2 sources principales (cf. consigne "1-2 sources principales") —
// jamais tronqué au milieu d'une phrase : seul l'extrait de chaque evidence
// est raccourci (à une limite de mots, pas de caractères bruts) si trop long.
//
// Réutilisée à la fois par VerdictCard (partage du verdict qui vient d'être
// rendu) et HistoryPanel (partage d'une entrée d'historique) : même format,
// une seule implémentation.
export function buildShareText(result: any): string {
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

// Partage via l'API native si disponible (mobile), sinon presse-papier.
// Retourne true si le presse-papier a été utilisé (pour afficher une
// confirmation "Copié !"), false si le partage natif a pris le relais.
export async function shareOrCopy(result: any): Promise<boolean> {
  const text = buildShareText(result);

  if ((navigator as any).share) {
    try {
      await (navigator as any).share({ text });
      return false;
    } catch (err) {
      // Partage annulé par l'utilisateur ou API refusée dans ce contexte :
      // on retombe silencieusement sur le presse-papier ci-dessous.
    }
  }

  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    alert("Impossible de copier automatiquement. Voici le texte à partager :\n\n" + text);
    return false;
  }
}
