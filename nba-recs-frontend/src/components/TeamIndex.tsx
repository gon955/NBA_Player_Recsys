"use client";

import { FORMER_TEAMS, teamColor } from "@/lib/teams";

type TeamIndexProps = {
  teams: string[];
  onSelect: (team: string) => void;
};

function TeamList({ teams, onSelect }: TeamIndexProps) {
  return (
    <ul className="columns-1 sm:columns-2 lg:columns-3 gap-10">
      {teams.map((t) => (
        <li key={t} className="break-inside-avoid border-t border-ink/25">
          <button
            type="button"
            onClick={() => onSelect(t)}
            className="group flex w-full items-center gap-3 py-2 text-left font-semibold cursor-pointer hover:bg-ink/5"
          >
            <span aria-hidden className="h-3 w-3 shrink-0" style={{ background: teamColor(t) }} />
            <span className="group-hover:underline">{t}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

/** What the page shows before a team is chosen: every team, one click away. */
export default function TeamIndex({ teams, onSelect }: TeamIndexProps) {
  const current = teams.filter((t) => !FORMER_TEAMS.has(t));
  const former = teams.filter((t) => FORMER_TEAMS.has(t));

  return (
    <section className="mt-10">
      <h2 className="text-2xl font-black tracking-tight">Pick a team</h2>
      <div className="mt-4">
        <TeamList teams={current} onSelect={onSelect} />
      </div>
      {former.length > 0 && (
        <>
          <h3 className="mt-10 text-sm font-bold text-muted">Former franchises</h3>
          <div className="mt-2">
            <TeamList teams={former} onSelect={onSelect} />
          </div>
        </>
      )}
    </section>
  );
}
