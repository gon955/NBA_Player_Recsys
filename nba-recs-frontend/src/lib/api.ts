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
  team_cluster_image?: string | null;
  player_cluster_image?: string | null;
  error?: string;
} | null;

export type Recommendation = {
  player: string;
  score: number;
  photo_url?: string | null;
  explanation?: RecommendationExplanation;
};

export type RecommendResponse = 
    | { user: string; era: string; recommendations: Recommendation[] }
    | { error: string};

export type ClusterSummaryRecord = {
  era: string;
  cluster_label: string;
  count: number;
};

export type ClusterSummaryResponse = {
  kind: "player" | "team";
  clusters: ClusterSummaryRecord[];
};

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export async function getTeams(): Promise<Record<string, number[]>> {
  const res = await fetch(`${BASE}/teams`);
  if (!res.ok) throw new Error(`Failed to fetch teams: ${res.status}`);
  return res.json();
}

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

async function fetchClusterSummary(kind: "player" | "team") {
  const res = await fetch(`${BASE}/cluster-summary/${kind}`);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Failed to fetch ${kind} clusters`);
  }
  const data = (await res.json()) as ClusterSummaryResponse;
  return data.clusters;
}

export async function getClusterSummaries() {
  const [player, team] = await Promise.all([
    fetchClusterSummary("player"),
    fetchClusterSummary("team"),
  ]);
  return { player, team };
}
