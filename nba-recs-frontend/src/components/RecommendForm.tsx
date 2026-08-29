"use client";

type RecommendFormProps = {
  teams: Record<string, number[]>;
  team: string;
  season: number | null;
  k: number;
  loading: boolean;
  error: string | null;
  canSubmit: boolean;
  onTeamChange: (team: string) => void;
  onSeasonChange: (season: number | null) => void;
  onKChange: (k: number) => void;
  onSubmit: (e: React.FormEvent) => void;
};

const selectClass =
  "rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-indigo-400 focus:outline-none";

export default function RecommendForm({
  teams,
  team,
  season,
  k,
  loading,
  error,
  canSubmit,
  onTeamChange,
  onSeasonChange,
  onKChange,
  onSubmit,
}: RecommendFormProps) {
  const teamNames = Object.keys(teams);

  return (
    <form
      onSubmit={onSubmit}
      className="mt-6 grid gap-4 rounded-2xl bg-slate-900/70 p-5 shadow-lg border border-white/5"
    >
      <div className="grid gap-1">
        <label className="text-sm font-medium text-slate-200" htmlFor="team">
          Team
        </label>
        <select
          id="team"
          className={selectClass}
          value={team}
          onChange={(e) => onTeamChange(e.target.value)}
        >
          <option value="">{teamNames.length ? "Select a team" : "Loading teams…"}</option>
          {teamNames.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>

      {team && (
        <div className="grid gap-1">
          <label className="text-sm font-medium text-slate-200" htmlFor="season">
            Season
          </label>
          <select
            id="season"
            className={selectClass}
            value={season ?? ""}
            onChange={(e) => onSeasonChange(e.target.value ? Number(e.target.value) : null)}
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

      <div className="grid gap-1">
        <label className="text-sm font-medium text-slate-200" htmlFor="k">
          Number of Recommendations
        </label>
        <input
          id="k"
          type="number"
          min={1}
          max={50}
          className={`w-32 ${selectClass}`}
          value={k}
          onChange={(e) => onKChange(Number(e.target.value))}
        />
      </div>

      <div className="flex gap-3 items-center">
        <button
          disabled={!canSubmit || loading}
          className="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-white font-semibold disabled:opacity-40"
        >
          {loading ? "Scoring…" : "Get Recommendations"}
        </button>
        {error && <span className="text-sm text-rose-400">{error}</span>}
      </div>
    </form>
  );
}
