// URL de base de l'API backend. Le port du backend peut varier d'une machine
// à l'autre (ex. 8000 déjà occupé par un autre service local) : configurez
// VITE_API_BASE_URL dans frontend/.env pour l'adapter, voir .env.example.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
