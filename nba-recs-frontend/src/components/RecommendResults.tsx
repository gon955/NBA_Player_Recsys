import type { Recommendation } from "@/lib/api";
import ExplanationCard from "./ExplanationCard";
import PlayerAvatar from "./PlayerAvatar";

type RecommendResultsProps = {
  loading: boolean;
  error: string | null;
  results: Recommendation[];
};

export default function RecommendResults({ loading, error, results }: RecommendResultsProps) {
  return (
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
                <div>
                  <p className="font-medium text-slate-100">
                    {idx + 1}. {r.player}
                  </p>
                  <p className="text-xs text-slate-400 font-mono">
                    score {r.score.toFixed(3)}
                  </p>
                </div>
              </div>
              {r.explanation && <ExplanationCard explanation={r.explanation} />}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
