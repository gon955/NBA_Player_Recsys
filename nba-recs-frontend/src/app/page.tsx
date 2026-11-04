"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AskPanel from "@/components/AskPanel";
import RecommendForm from "@/components/RecommendForm";
import RecommendResults from "@/components/RecommendResults";
import TeamIndex from "@/components/TeamIndex";
import ViewTabs, { type View } from "@/components/ViewTabs";
import {
  askQuestion,
  getRecommendations,
  getTeams,
  type Recommendation,
} from "@/lib/api";
import { INK, teamColor } from "@/lib/teams";

/** An aborted fetch is a superseded request, not a failure worth showing. */
function isAbort(err: unknown) {
  return err instanceof DOMException && err.name === "AbortError";
}

function message(err: unknown, fallback = "Something went wrong. Try again.") {
  // fetch rejects with a TypeError ("Failed to fetch") when it gets no response.
  if (err instanceof TypeError) return "Could not reach the server. Try again.";
  return (err as Error)?.message || fallback;
}

export default function HomePage() {
  const [view, setView] = useState<View>("recommend");

  const [teams, setTeams] = useState<Record<string, number[]>>({});
  const [teamsError, setTeamsError] = useState<string | null>(null);
  const [team, setTeam] = useState("");
  const [season, setSeason] = useState<number | null>(null);
  const [k, setK] = useState(10);
  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recsAbort = useRef<AbortController | null>(null);

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
        if (!isAbort(err)) setTeamsError("Could not load the team list. Try reloading the page.");
      });
    return () => ctrl.abort();
  }, []);

  // Abort anything still in flight when the page goes away.
  useEffect(
    () => () => {
      recsAbort.current?.abort();
      askAbort.current?.abort();
    },
    [],
  );

  const loadRecs = useCallback(async (nextTeam: string, nextSeason: number, nextK: number) => {
    recsAbort.current?.abort();
    const ctrl = new AbortController();
    recsAbort.current = ctrl;

    setLoading(true);
    setError(null);
    setResults([]);

    try {
      const data = await getRecommendations(
        { user_id: `${nextTeam}_${nextSeason}`, k: nextK },
        ctrl.signal,
      );
      setResults(data.recommendations);
    } catch (err) {
      if (!isAbort(err)) setError(message(err));
    } finally {
      if (!ctrl.signal.aborted) setLoading(false);
    }
  }, []);

  /** Choosing a team opens its latest season; an empty choice returns to the index. */
  function onTeamChange(next: string) {
    setTeam(next);
    const latest = teams[next]?.at(-1) ?? null;
    setSeason(latest);
    if (next && latest !== null) {
      loadRecs(next, latest, k);
    } else {
      recsAbort.current?.abort();
      setLoading(false);
      setError(null);
      setResults([]);
    }
  }

  function onSeasonChange(next: number) {
    setSeason(next);
    if (team) loadRecs(team, next, k);
  }

  function onKChange(next: number) {
    setK(next);
    if (team && season !== null) loadRecs(team, season, next);
  }

  async function onAsk(query: string) {
    if (!query.trim()) return;

    askAbort.current?.abort();
    const ctrl = new AbortController();
    askAbort.current = ctrl;

    setAskLoading(true);
    setAskError(null);
    setAskAnswer(null);

    try {
      setAskAnswer(await askQuestion(query, askEra || undefined, ctrl.signal));
    } catch (err) {
      if (!isAbort(err)) setAskError(message(err));
    } finally {
      if (!ctrl.signal.aborted) setAskLoading(false);
    }
  }

  const accent = team ? teamColor(team) : INK;

  return (
    <div style={{ "--team": accent } as React.CSSProperties} className="min-h-screen">
      <div className="mx-auto max-w-6xl px-4 py-6 md:px-6">
        <header>
          <h1 className="border-b-[6px] border-team pb-3 text-[clamp(2.25rem,7vw,5rem)] leading-[0.95] font-black tracking-[-0.03em]">
            NBA Player Recommender
          </h1>
          <div className="flex flex-wrap items-center justify-between gap-x-8 border-b border-ink">
            <ViewTabs view={view} onChange={setView} />
            <p className="py-2 font-serif text-base text-muted italic">
              Who would make a team better next season, among moves it could actually make.
            </p>
          </div>
        </header>

        <main>
          {view === "recommend" && (
            <>
              <RecommendForm
                teams={teams}
                team={team}
                season={season}
                k={k}
                onTeamChange={onTeamChange}
                onSeasonChange={onSeasonChange}
                onKChange={onKChange}
              />
              {teamsError && <p className="mt-8 font-semibold">{teamsError}</p>}
              {team && season !== null ? (
                <RecommendResults
                  team={team}
                  season={season}
                  loading={loading}
                  error={error}
                  results={results}
                  onClear={() => onTeamChange("")}
                />
              ) : (
                !teamsError && <TeamIndex teams={Object.keys(teams)} onSelect={onTeamChange} />
              )}
            </>
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
              onAsk={onAsk}
            />
          )}
        </main>

        <footer className="mt-16 border-t-2 border-ink pt-3">
          <h2 className="text-sm font-black">How a list is made</h2>
          <div className="mt-2 grid gap-x-8 gap-y-3 font-serif text-[0.95rem] leading-snug text-muted md:grid-cols-3">
            <p>
              <strong className="font-semibold text-ink">Value.</strong> A player&apos;s projected
              next-season impact, from a possession-weighted model of five-man lineups, adjusted for
              how players do after changing teams.
            </p>
            <p>
              <strong className="font-semibold text-ink">Realism.</strong> How likely this team is
              to land him, learned from real arrivals in earlier seasons, among trades that work
              under salary-matching rules.
            </p>
            <p>
              <strong className="font-semibold text-ink">Fit.</strong> How weak the team is at his
              position, because players joining a thin position get more minutes.
            </p>
          </div>
          <p className="mt-4 text-xs text-muted">
            Data from Basketball-Reference and pbpstats.{" "}
            <a
              className="underline hover:text-ink"
              href="https://github.com/gon955/NBA_Player_Recsys"
            >
              Source and methods on GitHub
            </a>
            .
          </p>
        </footer>
      </div>
    </div>
  );
}
