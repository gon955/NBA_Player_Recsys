export type RecommendPayload = {
  user_id: string;
  k?: number;
};

/** Why a player is on the list: the move the impact recommender found.
 *  `value` is projected net impact next season if he changes teams, points
 *  per 100 possessions vs. league average; `position_value` the same for the
 *  team's current rotation at his position, minutes-weighted; `minutes` his floor
 *  time last season per team game. `replaces` and `with` are the contracts going
 *  out to make the trade legal, not necessarily whose minutes he would take. */
export type RecommendationExplanation = {
  age: number | null;
  bpm: number | null;
  value: number | null;
  position_value: number | null;
  replaces: string;
  replaces_id: string;
  with: string[];
  with_salary: number | null;
  salary: number | null;
  replaces_salary: number | null;
  cap: number | null;
  minutes: number | null;
} | null;

export type Recommendation = {
  player: string;
  score: number;
  photo_url?: string | null;
  explanation?: RecommendationExplanation;
};

export type RecommendResponse =
  | { user: string; era: string | null; recommendations: Recommendation[] }
  | { error: string };

/** Single source of truth for the backend origin. Baked at build time, so a
 *  containerized frontend must be rebuilt to point at a different backend. */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

async function failure(res: Response, fallback: string) {
  const text = await res.text().catch(() => "");
  return new Error(text || fallback);
}

export async function getTeams(signal?: AbortSignal): Promise<Record<string, number[]>> {
  const res = await fetch(`${API_BASE}/teams`, { signal });
  if (!res.ok) throw await failure(res, `Failed to fetch teams: ${res.status}`);
  return res.json();
}

export async function getRecommendations(payload: RecommendPayload, signal?: AbortSignal) {
  const res = await fetch(`${API_BASE}/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) throw await failure(res, `Request failed with status ${res.status}`);

  // The backend reports domain errors (an unknown team-season) as HTTP 200
  // with an {error} body, so status alone is not enough to detect failure.
  const data = (await res.json()) as RecommendResponse;
  if ("error" in data) throw new Error(data.error);
  return data;
}

export async function askQuestion(
  query: string,
  era?: string,
  signal?: AbortSignal,
): Promise<string> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, era: era || null, n_results: 10 }),
    signal,
  });
  if (!res.ok) throw await failure(res, "Failed to get answer");
  const data = (await res.json()) as { answer?: string; error?: string };
  if (data.error) throw new Error(data.error);
  return data.answer ?? "";
}
