import type { PlayerProjection, TradeRecommendation } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  return r.json() as Promise<T>;
}

export async function fetchPlayers(): Promise<PlayerProjection[]> {
  return j<PlayerProjection[]>("/players");
}

export async function evaluateTeam(playerIds: number[]) {
  return j<{ expected_total: number; ci_low: number; ci_high: number; per_player: PlayerProjection[] }>(
    "/team-evaluate",
    { method: "POST", body: JSON.stringify({ player_ids: playerIds }) },
  );
}

export async function recommendTrades(body: {
  field_player_ids: number[];
  bank: number;
  max_trades?: number;
  horizon_rounds?: number;
  mc_samples?: number;
}): Promise<TradeRecommendation[]> {
  return j<TradeRecommendation[]>("/trade-recommend", {
    method: "POST",
    body: JSON.stringify({
      max_trades: 2,
      horizon_rounds: 3,
      mc_samples: 1500,
      ...body,
    }),
  });
}
