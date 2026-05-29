"use client";

import { useEffect,useMemo, useState } from "react";
import { getRecommendations, getTeams, Recommendation, FeatureContribution, getClusterSummaries, ClusterSummaryRecord } from "@/lib/api";
import { askQuestion } from "@/lib/api";
function getEraFromYear(year: number): string {
  if (year >= 2016) return "2016-present";
  if (year >= 2008) return "2008-2015";
  return "1999-2007";
}

export default function HomePage() {
  const [teams, setTeams] = useState<Record<string, number[]>>({});
  const [team, setTeam] = useState<string>("");
  const [season, setSeason] = useState<number | null>(null);
  const [era, setEra] = useState<string>("2016-present");
  const [k, setK] = useState<number>(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Recommendation[]>([]);
  const [view, setView] = useState<"recommend" | "clusters" | "ask">("recommend");
  const [clusterData, setClusterData] = useState<{ player?: ClusterSummaryRecord[]; team?: ClusterSummaryRecord[] }>({});
  const [clusterError, setClusterError] = useState<string | null>(null);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [askQuery, setAskQuery] = useState<string>("");
  const [askAnswer, setAskAnswer] = useState<string | null>(null);
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  useEffect(() => {
    getTeams()
      .then((data) => setTeams(data))
      .catch(() => setError("Failed to load team list"));
  }, []);

  useEffect(() => {
    if (view !== "clusters" || clusterData.player) return;
    setClusterLoading(true);
    getClusterSummaries()
      .then((data) => setClusterData(data))
      .catch((err) => setClusterError(err?.message || "Failed to load clusters"))
      .finally(() => setClusterLoading(false));
  }, [view, clusterData.player]);
  const canSubmit = useMemo(() => !!team && !!season && k > 0, [team, season, k]);
  const userKey = team && season ? `${team}_${season}` : "";

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setResults([]);

    const eraVal = getEraFromYear(season!);
    setEra(eraVal);
    ///const ctrl = new AbortController();

    try {
      const data = await getRecommendations({ era: eraVal, user_id: userKey, k });
      setResults(data.recommendations);
    } catch (err: any) {
      setError(err?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }
  async function onAsk(e: React.FormEvent) {
  e.preventDefault();
  if (!askQuery.trim()) return;
  setAskLoading(true);
  setAskError(null);
  setAskAnswer(null);

  try {
    const answer = await askQuestion(askQuery, era || undefined);
    setAskAnswer(answer);
  } catch (err: any) {
    setAskError(err?.message || "Unknown error");
  } finally {
    setAskLoading(false);
  }
}

   return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="max-w-4xl mx-auto px-4 py-10">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">
          NBA Player Recommender
        </h1>
        <p className="text-slate-400 mt-2">
          LightFM recommendations served by FastAPI.
        </p>

        <div className="mt-6 flex gap-3">
          <button
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${view === "recommend" ? "bg-indigo-500 text-white" : "bg-slate-800 text-slate-300"}`}
            onClick={() => setView("recommend")}
          >
            Recommendations
          </button>
          <button
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${view === "clusters" ? "bg-indigo-500 text-white" : "bg-slate-800 text-slate-300"}`}
            onClick={() => setView("clusters")}
          >
            Cluster Overview
          </button>

          <button
            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${view === "ask" ? "bg-indigo-500 text-white" : "bg-slate-800 text-slate-300"}`}
            onClick={() => setView("ask")}
          >
            Ask Question
          </button>
        </div>

        {view === "recommend" && (
        <form
          onSubmit={onSubmit}
          className="mt-6 grid gap-4 rounded-2xl bg-slate-900/70 p-5 shadow-lg border border-white/5"
        >
          {/* Team Selector */}
          <div className="grid gap-1">
            <label className="text-sm font-medium text-slate-200">Team</label>
            <select
              className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-indigo-400 focus:outline-none"
              value={team}
              onChange={(e) => {
                setTeam(e.target.value);
                setSeason(null);
              }}
            >
              <option value="">Select a team</option>
              {Object.keys(teams).map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          {/* Season Selector */}
          {team && (
            <div className="grid gap-1">
              <label className="text-sm font-medium text-slate-200">Season</label>
              <select
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-indigo-400 focus:outline-none"
                value={season ?? ""}
                onChange={(e) => setSeason(Number(e.target.value))}
              >
                <option value="">Select a season</option>
                {teams[team]?.map((y) => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Number of Recommendations */}
          <div className="grid gap-1">
            <label className="text-sm font-medium text-slate-200">Number of Recommendations</label>
            <input
              type="number"
              min={1}
              className="w-32 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-indigo-400 focus:outline-none"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </div>

          <div className="flex gap-3">
            <button
              disabled={!canSubmit || loading}
              className="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-white font-semibold disabled:opacity-40"
            >
              {loading ? "Scoring…" : "Get Recommendations"}
            </button>
            {error && <span className="text-sm text-rose-400">{error}</span>}
          </div>
        </form>
        )}

        {view === "recommend" ? (
        <section className="mt-8">
          <h2 className="text-xl font-semibold text-slate-100">Results</h2>
          {loading && <p className="mt-2 text-slate-400">Scoring candidates…</p>}
          {!loading && results.length === 0 && !error && (
            <p className="mt-2 text-slate-500">No results yet. Try requesting recommendations.</p>
          )}
          {!loading && results.length > 0 && (
            <ul className="mt-4 grid gap-2">
              {results.map((r, idx) => (
                <li
                  key={`${r.player}-${idx}`}
                  className="rounded-2xl border border-white/5 bg-slate-900/70 px-4 py-4 shadow-lg"
                >
                  <div className="flex items-center gap-3">
                    <PlayerAvatar name={r.player} photoUrl={r.photo_url} />
                    <p className="font-medium text-slate-100">{idx + 1}. {r.player}</p>
                  </div>
                  {r.explanation && <ExplanationCard explanation={r.explanation} />}
                </li>
              ))}
            </ul>
          )}
        </section>
        ) : view === "clusters" ? (
          <ClusterOverview
            loading={clusterLoading}
            error={clusterError}
            data={clusterData}
          />
        ) : (
          <section className="mt-8">
            <h2 className="text-xl font-semibold text-slate-100">Ask About the Data</h2>
            <p className="text-slate-400 text-sm mt-1">
              Ask natural language questions about players, teams, and archetypes.
            </p>
            <form
              onSubmit={onAsk}
              className="mt-4 grid gap-4 rounded-2xl bg-slate-900/70 p-5 shadow-lg border border-white/5"
            >
              <div className="grid gap-1">
                <label className="text-sm font-medium text-slate-200">Your question</label>
                <textarea
                  className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-indigo-400 focus:outline-none resize-none"
                  rows={3}
                  placeholder="e.g. Who were the most efficient big men in the 1999-2007 era?"
                  value={askQuery}
                  onChange={(e) => setAskQuery(e.target.value)}
                />
              </div>
              <div className="flex gap-3 items-center">
                <button
                  disabled={!askQuery.trim() || askLoading}
                  className="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-white font-semibold disabled:opacity-40"
                >
                  {askLoading ? "Thinking…" : "Ask"}
                </button>
                {askError && <span className="text-sm text-rose-400">{askError}</span>}
              </div>
            </form>
            {askAnswer && (
              <div className="mt-6 rounded-2xl border border-white/5 bg-slate-900/70 px-5 py-4 shadow-lg">
                <p className="text-sm font-semibold text-slate-300 mb-2">Answer</p>
                <p className="text-slate-100 whitespace-pre-wrap leading-relaxed">{askAnswer}</p>
              </div>
            )}
          </section>
        )}
      </div>
    </main>
  );
}

type FeatureListProps = {
  title: string;
  items?: FeatureContribution[];
  variant: "user" | "item";
};

function FeatureList({ title, items, variant }: FeatureListProps) {
  if (!items || items.length === 0) return null;
  return (
    <div>
      <p className="text-sm font-semibold text-slate-200">{title}</p>
      <ul className="mt-1 flex flex-col gap-2 text-sm text-slate-200">
        {items.map((feat) => (
          <li
            key={`${title}-${feat.feature}`}
            className="rounded-lg bg-slate-900/80 px-3 py-2 ring-1 ring-white/10"
          >
            <p className="font-medium text-slate-100">
              {describeFeature(feat.feature, feat.weight, variant)}
            </p>
            <p className="text-xs text-slate-400">
              Impact: {feat.weight >= 0 ? "+" : ""}
              {feat.weight.toFixed(2)}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

type ExplanationCardProps = {
  explanation: Recommendation["explanation"];
};

function ExplanationCard({ explanation }: ExplanationCardProps) {
  if (!explanation) return null;
  const { error, top_user_features, top_item_features, team_cluster_image, player_cluster_image } = explanation;

  return (
    <div className="mt-3 rounded-xl bg-slate-900/80 px-4 py-3 text-sm text-slate-200 ring-1 ring-white/10">
      {error ? (
        <p className="text-rose-400">Explanation unavailable: {error}</p>
      ) : (
        <div className="grid gap-6 sm:grid-cols-2">
          <div className="space-y-3">
            <FeatureList title="Team fit drivers" items={top_user_features} variant="user" />
            {team_cluster_image && (
              <ClusterImage label="Team cluster" src={team_cluster_image} />
            )}
          </div>
          <div className="space-y-3">
            <FeatureList title="Player traits" items={top_item_features} variant="item" />
            {player_cluster_image && (
              <ClusterImage label="Player cluster" src={player_cluster_image} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

type PlayerAvatarProps = {
  name: string;
  photoUrl?: string | null;
};

function PlayerAvatar({ name, photoUrl }: PlayerAvatarProps) {
  const [failed, setFailed] = useState(false);
  const initials = name
    .split(" ")
    .filter(Boolean)
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  return (
    <div className="h-16 w-16 flex items-center justify-center rounded-2xl bg-slate-800 ring-1 ring-white/10 overflow-hidden">
      {photoUrl && !failed ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={photoUrl}
          alt={`${name} headshot`}
          className="h-full w-full object-contain bg-white"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="text-sm font-semibold text-gray-600">{initials}</span>
      )}
    </div>
  );
}

type ClusterImageProps = {
  label: string;
  src: string;
};

function ClusterImage({ label, src }: ClusterImageProps) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-slate-400 mb-1">{label}</p>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={label}
        className="w-full rounded-lg border border-white/10 bg-slate-900 object-contain"
      />
    </div>
  );
}

type ClusterOverviewProps = {
  loading: boolean;
  error: string | null;
  data: { player?: ClusterSummaryRecord[]; team?: ClusterSummaryRecord[] };
};

function ClusterOverview({ loading, error, data }: ClusterOverviewProps) {
  return (
    <section className="mt-8 space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-slate-100">Cluster Overview</h2>
        <p className="text-slate-400 text-sm mt-1">
          Dominant clusters per era with population counts.
        </p>
      </div>
      {loading && <p className="text-slate-400">Loading cluster data…</p>}
      {error && <p className="text-rose-400">{error}</p>}
      {!loading && !error && (
        <div className="space-y-6">
          <ClusterSummaryCard
            title="Player clusters"
            kind="player"
            records={data.player}
          />
          <ClusterSummaryCard
            title="Team clusters"
            kind="team"
            records={data.team}
          />
        </div>
      )}
    </section>
  );
}

type ClusterSummaryCardProps = {
  title: string;
  kind: "player" | "team";
  records?: ClusterSummaryRecord[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

function ClusterSummaryCard({ title, kind, records }: ClusterSummaryCardProps) {
  if (!records || records.length === 0) {
    return (
      <div className="rounded-2xl border border-white/5 bg-slate-900/70 px-4 py-4 shadow-lg">
        <p className="text-slate-300 text-sm">{title}: no data</p>
      </div>
    );
  }
  const maxCount = Math.max(...records.map((r) => r.count));
  const eras = Array.from(new Set(records.map((r) => r.era)));

  const imageMap: Record<string, string> = {};
  eras.forEach((era) => {
    const slug = era.replace(/ /g, "_");
    imageMap[era] = `${API_BASE}/static/cluster/${kind === "player" ? "player" : "team"}_pca_${slug}.png`;
  });

  return (
    <div className="rounded-2xl border border-white/5 bg-slate-900/70 px-4 py-4 shadow-lg space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-100">{title}</h3>
      </div>
      {eras.map((era) => {
        const eraClusters = records.filter((r) => r.era === era);
        return (
          <div key={era} className="space-y-3">
            <div className="flex items-center justify-between">
              <p className="font-semibold text-slate-200">{era}</p>
              {imageMap[era] && (
                <a
                  href={imageMap[era]}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-indigo-300 hover:text-indigo-200"
                >
                  View PCA
                </a>
              )}
            </div>
            <div className="space-y-2">
              {eraClusters.map((cluster) => (
                <div key={cluster.cluster_label} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span>{cluster.cluster_label}</span>
                    <span>{cluster.count}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-indigo-500"
                      style={{ width: `${(cluster.count / maxCount) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function describeFeature(feature: string, weight: number, variant: "user" | "item") {
  const [rawKey, rawValue = ""] = feature.split("=");
  const key = rawKey?.trim() ?? "";
  const value = formatValueByKey(key, rawValue);
  const sentiment = weight >= 0 ? "Boosts fit" : "Adds risk";

  const userDescriptions: Record<string, string> = {
    tcluster: `Team identity is ${value}`,
    pace: `Team pace target is ${value}`,
    ortg: `Offensive profile leans ${value}`,
    drtg: `Defensive profile trends ${value}`,
    era: `Team era tag is ${value}`,
  };

  const itemDescriptions: Record<string, string> = {
    pcluster: `Player archetype: ${value}`,
    pos: `Primary position: ${value}`,
    age: `Age band: ${value}`,
    era: `Season belongs to ${value}`,
  };

  if (variant === "user" && userDescriptions[key]) {
    return `${sentiment} because ${userDescriptions[key]}`;
  }
  if (variant === "item" && itemDescriptions[key]) {
    return `${sentiment} thanks to ${itemDescriptions[key]}`;
  }
  const fallback = value || key || feature;
  return `${sentiment} via ${formatValue(fallback)}`;
}

function formatValueByKey(key: string, rawValue: string) {
  const valueMaps: Record<string, Record<string, string>> = {
    pace: {
      slow: "Slow Pace",
      "med-": "Moderately Slow Pace",
      "med+": "Moderately Fast Pace",
      fast: "Fast Pace",
    },
    ortg: {
      o_low: "Low Offense",
      "o_mid-": "Below-average Offense",
      "o_mid+": "Above-average Offense",
      o_high: "High Offense",
    },
    drtg: {
      d_best: "Elite Defense",
      d_good: "Good Defense",
      d_ok: "Average Defense",
      d_poor: "Poor Defense",
    },
    age: {
      U22: "22 or younger",
      "23-26": "Ages 23-26",
      "27-30": "Ages 27-30",
      "31-34": "Ages 31-34",
      "35-39": "Ages 35-39",
      "40+": "40+",
    },
  };

  const map = valueMaps[key];
  if (map && rawValue in map) {
    return map[rawValue];
  }
  return formatValue(rawValue);
}

function formatValue(value: string) {
  if (!value) return "unknown";
  return value.replace(/_/g, " ");
}

