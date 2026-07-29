// Identifiant utilisateur léger : généré une fois côté client et conservé
// dans localStorage, SANS mot de passe ni compte. Sert uniquement à
// retrouver son propre historique sur ce même navigateur (voir
// DOCUMENTATION_TECHNIQUE.md pour les limites assumées de ce modèle).
const STORAGE_KEY = 'terrava_user_id';

export function getUserId(): string {
  let id = localStorage.getItem(STORAGE_KEY);
  if (!id) {
    id = typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `user-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    localStorage.setItem(STORAGE_KEY, id);
  }
  return id;
}
