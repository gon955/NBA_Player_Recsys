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
  | { error: string };

export type ClusterSummaryRecord = {
  era: string;
  cluster_label: string;
  count: number;
};

export type ClusterSummaryResponse = {
  kind: "player" | "team";
  clusters: ClusterSummaryRecord[];
};

export type ClusterKind = "player" | "team";

/** Single source of truth for the backend origin. Baked at build time, so a
 *  containerized frontend must be rebuilt to point at a different backend. */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

/** Cluster PCA scatter plots are served straight off the backend's /static mount. */
export function clusterPcaUrl(kind: ClusterKind, era: string) {
  return `${API_BASE}/static/cluster/${kind}_pca_${era.replace(/ /g, "_")}.png`;
}

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

  // The backend reports domain errors (unknown era, missing user) as HTTP 200
  // with an {error} body, so status alone is not enough to detect failure.
  const data = (await res.json()) as RecommendResponse;
  if ("error" in data) throw new Error(data.error);
  return data;
}

async function fetchClusterSummary(kind: ClusterKind, signal?: AbortSignal) {
  const res = await fetch(`${API_BASE}/cluster-summary/${kind}`, { signal });
  if (!res.ok) throw await failure(res, `Failed to fetch ${kind} clusters`);
  const data = (await res.json()) as ClusterSummaryResponse;
  return data.clusters;
}

export async function getClusterSummaries(signal?: AbortSignal) {
  const [player, team] = await Promise.all([
    fetchClusterSummary("player", signal),
    fetchClusterSummary("team", signal),
  ]);
  return { player, team };
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
