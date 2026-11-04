"use client";

export const LIST_SIZES = [10, 25, 50] as const;

type RecommendFormProps = {
  teams: Record<string, number[]>;
  team: string;
  season: number | null;
  k: number;
  onTeamChange: (team: string) => void;
  onSeasonChange: (season: number) => void;
  onKChange: (k: number) => void;
};

/** The list loads as soon as any of these change, so there is no submit button. */
export default function RecommendForm({
  teams,
  team,
  season,
  k,
  onTeamChange,
  onSeasonChange,
  onKChange,
}: RecommendFormProps) {
  const teamNames = Object.keys(teams);
  const seasons = [...(teams[team] ?? [])].reverse();

  return (
    <div className="mt-6 flex flex-wrap items-end gap-x-8 gap-y-4">
      <label className="grid gap-1 text-sm font-semibold text-muted">
        Team
        <select
          className="select w-64 max-w-full text-lg text-ink"
          value={team}
          onChange={(e) => onTeamChange(e.target.value)}
        >
          <option value="">{teamNames.length ? "All teams" : "Loading teams…"}</option>
          {teamNames.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>

      <label className="grid gap-1 text-sm font-semibold text-muted">
        After season
        <select
          className="select w-28 text-lg text-ink tabular-nums disabled:opacity-40"
          disabled={!team}
          value={season ?? ""}
          onChange={(e) => onSeasonChange(Number(e.target.value))}
        >
          {!team && <option value="">—</option>}
          {seasons.map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>
      </label>

      <fieldset className="grid gap-1">
        <legend className="mb-1 text-sm font-semibold text-muted">Show</legend>
        <div className="flex border-2 border-ink">
          {LIST_SIZES.map((n) => (
            <label
              key={n}
              className={`cursor-pointer px-3 py-[0.3rem] font-bold tabular-nums has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-team ${
                k === n ? "bg-ink text-paper" : "hover:bg-ink/10"
              }`}
            >
              <input
                type="radio"
                name="list-size"
                className="sr-only"
                checked={k === n}
                onChange={() => onKChange(n)}
              />
              {n}
            </label>
          ))}
        </div>
      </fieldset>
    </div>
  );
}
