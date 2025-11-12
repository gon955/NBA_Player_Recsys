"use client";

import { useEffect,useMemo, useState } from "react";
import { getRecommendations, Recommendation } from "@/lib/api";

function getEraFromYear(year: number): string {
  if (year >= 2016) return "2016-present";
  if (year >= 2008) return "2008-2015";
  return "1999-2007";
}

const ERAS = ["1999-2007", "2008-2015", "2016-present"];

export default function HomePage() {
  const [teams, setTeams] = useState<Record<string, number[]>>({});
  const [team, setTeam] = useState<string>("");
  const [season, setSeason] = useState<number | null>(null);
  const [era, setEra] = useState<string>("2016-present");
  const [userId, setUserId] = useState<string>("Atlanta Hawks_2019");
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

  const canSubmit = useMemo(() => !!era && !!userId && k > 0, [era, userId, k]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setResults([]);

    const userId = `${team}_${season}`;
    const eraVal = getEraFromYear(season!);
    setEra(eraVal);
    ///const ctrl = new AbortController();

    try {
      const data = await getRecommendations({ era, user_id: userId, k },);
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
                  className="flex items-center justify-between rounded-xl border bg-white px-4 py-3"
                >
                  <span className="font-medium">{idx + 1}. {r.player}</span>
                  <span className="tabular-nums text-gray-600">{r.score.toFixed(4)}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </main>
  );
}