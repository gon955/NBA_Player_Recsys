export type RecommendPayload = {
    era: string;
    user_id: string;
    k?: number;
};

export type FeatureContribution = {
  feature: string;
  weight: number;
};

export type RecommendationExplanation = {
  score_total?: number;
  user_bias?: number;
  item_bias?: number;
  top_user_features?: FeatureContribution[];
  top_item_features?: FeatureContribution[];
  error?: string;
} | null;

export type Recommendation = {
  player: string;
  score: number;
  explanation?: RecommendationExplanation;
};

export type RecommendResponse = 
    | { user: string; era: string; recommendations: Recommendation[] }
    | { error: string};

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export async function getRecommendations(payload: RecommendPayload, signal?: AbortSignal) {
  const res = await fetch(`${BASE}/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed with status ${res.status}`);
  }

  const data = (await res.json()) as RecommendResponse;
  if ("error" in data) throw new Error(data.error);
  return data;
}
