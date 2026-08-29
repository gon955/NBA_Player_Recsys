"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AskPanel from "@/components/AskPanel";
import ClusterOverview from "@/components/ClusterOverview";
import RecommendForm from "@/components/RecommendForm";
import RecommendResults from "@/components/RecommendResults";
import ViewTabs, { type View } from "@/components/ViewTabs";
import {
  askQuestion,
  getClusterSummaries,
  getRecommendations,
  getTeams,
  type ClusterSummaryRecord,
  type Recommendation,
} from "@/lib/api";
import { getEraFromYear } from "@/lib/era";

/** An aborted fetch is a superseded request, not a failure worth showing. */
function isAbort(err: unknown) {
  return err instanceof DOMException && err.name === "AbortError";
}

function message(err: unknown, fallback = "Unknown error") {
  return (err as Error)?.message || fallback;
}

export default function HomePage() {
  const [view, setView] = useState<View>("recommend");

  const [teams, setTeams] = useState<Record<string, number[]>>({});
  const [team, setTeam] = useState("");
  const [season, setSeason] = useState<number | null>(null);
  const [k, setK] = useState(10);
  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recsAbort = useRef<AbortController | null>(null);

  const [clusterData, setClusterData] = useState<{
    player?: ClusterSummaryRecord[];
    team?: ClusterSummaryRecord[];
  }>({});
  const [clusterLoading, setClusterLoading] = useState(false);
  const [clusterError, setClusterError] = useState<string | null>(null);

  const [askQuery, setAskQuery] = useState("");
  const [askEra, setAskEra] = useState("");
  const [askAnswer, setAskAnswer] = useState<string | null>(null);
  const [askLoading, setAskLoading] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const askAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    getTeams(ctrl.signal)
      .then(setTeams)
      .catch((err) => {
        if (!isAbort(err)) setError("Failed to load team list");
      });
    return () => ctrl.abort();
  }, []);

  // Cluster summaries are static, so fetch them once on first visit to the tab.
  useEffect(() => {
    if (view !== "clusters" || clusterData.player) return;
    const ctrl = new AbortController();
    setClusterLoading(true);
    setClusterError(null);
    getClusterSummaries(ctrl.signal)
      .then(setClusterData)
      .catch((err) => {
        if (!isAbort(err)) setClusterError(message(err, "Failed to load clusters"));
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setClusterLoading(false);
      });
    return () => ctrl.abort();
  }, [view, clusterData.player]);

  // Abort anything still in flight when the page goes away.
  useEffect(
    () => () => {
      recsAbort.current?.abort();
      askAbort.current?.abort();
    },
    [],
  );

  const canSubmit = useMemo(() => !!team && !!season && k > 0, [team, season, k]);

  const onTeamChange = useCallback((next: string) => {
    setTeam(next);
    setSeason(null);
  }, []);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || season === null) return;

    recsAbort.current?.abort();
    const ctrl = new AbortController();
    recsAbort.current = ctrl;

    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const data = await getRecommendations(
        { era: getEraFromYear(season), user_id: `${team}_${season}`, k },
        ctrl.signal,
      );
      setResults(data.recommendations);
    } catch (err) {
      if (!isAbort(err)) setError(message(err));
    } finally {
      if (!ctrl.signal.aborted) setLoading(false);
    }
  }

  async function onAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!askQuery.trim()) return;

    askAbort.current?.abort();
    const ctrl = new AbortController();
    askAbort.current = ctrl;

    setAskLoading(true);
    setAskError(null);
    setAskAnswer(null);

    try {
      setAskAnswer(await askQuestion(askQuery, askEra || undefined, ctrl.signal));
    } catch (err) {
      if (!isAbort(err)) setAskError(message(err));
    } finally {
      if (!ctrl.signal.aborted) setAskLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="max-w-4xl mx-auto px-4 py-10">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">
          NBA Player Recommender
        </h1>
        <p className="text-slate-400 mt-2">LightFM recommendations served by FastAPI.</p>

        <ViewTabs view={view} onChange={setView} />

        {view === "recommend" && (
          <>
            <RecommendForm
              teams={teams}
              team={team}
              season={season}
              k={k}
              loading={loading}
              error={error}
              canSubmit={canSubmit}
              onTeamChange={onTeamChange}
              onSeasonChange={setSeason}
              onKChange={setK}
              onSubmit={onSubmit}
            />
            <RecommendResults loading={loading} error={error} results={results} />
          </>
        )}

        {view === "clusters" && (
          <ClusterOverview loading={clusterLoading} error={clusterError} data={clusterData} />
        )}

        {view === "ask" && (
          <AskPanel
            query={askQuery}
            era={askEra}
            answer={askAnswer}
            loading={askLoading}
            error={askError}
            onQueryChange={setAskQuery}
            onEraChange={setAskEra}
            onSubmit={onAsk}
          />
        )}
      </div>
    </main>
  );
}
