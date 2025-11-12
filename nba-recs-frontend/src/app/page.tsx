"use client";

import { useEffect,useMemo, useState } from "react";
import { getRecommendations, Recommendation, FeatureContribution } from "@/lib/api";

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

  useEffect(() => {
    fetch("http://127.0.0.1:8000/teams")
      .then((res) => res.json())
      .then((data) => setTeams(data))
      .catch(() => setError("Failed to load team list"));
  }, []);

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

   return (
    <main className="min-h-screen bg-gray-50 text-gray-900">
      <div className="max-w-4xl mx-auto px-4 py-10">
        <h1 className="text-3xl font-bold tracking-tight text-indigo-700">
          NBA Player Recommender
        </h1>
        <p className="text-gray-600 mt-2">
          LightFM recommendations served by FastAPI.
        </p>

        <form
          onSubmit={onSubmit}
          className="mt-6 grid gap-4 rounded-2xl bg-white p-5 shadow-sm"
        >
          {/* Team Selector */}
          <div className="grid gap-1">
            <label className="text-sm font-medium">Team</label>
            <select
              className="rounded-lg border border-gray-600 px-3 py-2 text-gray-900"
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
              <label className="text-sm font-medium">Season</label>
              <select
                className="rounded-lg border border-gray-600 px-3 py-2 text-gray-900"
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
            <label className="text-sm font-medium">Number of Recommendations</label>
            <input
              type="number"
              min={1}
              className="w-32 rounded-lg border border-gray-600 px-3 py-2"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </div>

          <div className="flex gap-3">
            <button
              disabled={!canSubmit || loading}
              className="rounded-xl bg-indigo-600 hover:bg-indigo-700 px-4 py-2 text-white font-semibold disabled:opacity-50"
            >
              {loading ? "Scoring…" : "Get Recommendations"}
            </button>
            {error && <span className="text-sm text-red-600">{error}</span>}
          </div>
        </form>

        <section className="mt-8">
          <h2 className="text-xl font-semibold text-gray-900">Results</h2>
          {loading && <p className="mt-2 text-gray-600">Scoring candidates…</p>}
          {!loading && results.length === 0 && !error && (
            <p className="mt-2 text-gray-600">No results yet. Try requesting recommendations.</p>
          )}
          {!loading && results.length > 0 && (
            <ul className="mt-4 grid gap-2">
              {results.map((r, idx) => (
                <li
                  key={`${r.player}-${idx}`}
                  className="rounded-xl border bg-white px-4 py-3"
                >
                  <div className="flex items-center gap-3">
                    <PlayerAvatar name={r.player} photoUrl={r.photo_url} />
                    <p className="font-medium text-gray-900">{idx + 1}. {r.player}</p>
                  </div>
                  {r.explanation && <ExplanationCard explanation={r.explanation} />}
                </li>
              ))}
            </ul>
          )}
        </section>
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
      <p className="text-sm font-semibold text-gray-900">{title}</p>
      <ul className="mt-1 flex flex-col gap-2 text-sm text-gray-700">
        {items.map((feat) => (
          <li
            key={`${title}-${feat.feature}`}
            className="rounded-lg bg-white/70 px-3 py-2 shadow-sm"
          >
            <p className="font-medium text-gray-900">
              {describeFeature(feat.feature, feat.weight, variant)}
            </p>
            <p className="text-xs text-gray-500">
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
  const { error, top_user_features, top_item_features } = explanation;

  return (
    <div className="mt-3 rounded-lg bg-gray-50 px-3 py-2 text-sm text-gray-700">
      {error ? (
        <p className="text-red-600">Explanation unavailable: {error}</p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          <FeatureList title="Team fit drivers" items={top_user_features} variant="user" />
          <FeatureList title="Player traits" items={top_item_features} variant="item" />
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
    <div className="h-16 w-16 flex items-center justify-center rounded-xl bg-gray-100 ring-1 ring-gray-200 overflow-hidden">
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
