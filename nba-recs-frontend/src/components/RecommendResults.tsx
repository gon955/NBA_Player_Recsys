import type { Recommendation } from "@/lib/api";
import PlayerAvatar from "./PlayerAvatar";

function money(x: number | null) {
  if (x === null) return "—";
  return x >= 1e6 ? `$${(x / 1e6).toFixed(1)}M` : `$${Math.round(x / 1e3)}K`;
}

function pts(x: number | null) {
  if (x === null) return "—";
  // Round before choosing the sign, so -0.04 reads +0.0 and not a minus zero.
  const rounded = Math.round(x * 10) / 10;
  return `${rounded >= 0 ? "+" : "\u2212"}${Math.abs(rounded).toFixed(1)}`;
}

/** Rank, photo, player, then the figures. Shared by the header and every row. */
const COLUMNS =
  "md:grid md:grid-cols-[2.25rem_3.5rem_minmax(0,1fr)_2.5rem_3.5rem_3.5rem_4.5rem_5.5rem_5.5rem] md:gap-x-4";

function Figure({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="md:text-right md:self-center">
      <span className="block text-xs font-semibold text-muted md:sr-only">{label}</span>
      <span className="tabular-nums">{children}</span>
    </div>
  );
}

function Row({ rec, rank }: { rec: Recommendation; rank: number }) {
  const e = rec.explanation;
  return (
    <li className="border-t border-ink/30 py-3">
      <div className={`${COLUMNS} flex flex-wrap items-center gap-x-3`}>
        <span className="w-7 text-lg font-black tabular-nums md:w-auto md:self-center md:text-right">
          {rank}
        </span>
        <PlayerAvatar name={rec.player} photoUrl={rec.photo_url} />
        <div className="min-w-0 flex-1 md:self-center">
          <p className="text-lg leading-tight font-extrabold tracking-tight">{rec.player}</p>
          {e && (
            <p className="mt-1 text-sm leading-snug text-muted">
              <span className="font-semibold text-ink">Salary going out: </span>
              {e.replaces} ({money(e.replaces_salary)})
              {e.with.length > 0 && (
                <>
                  {" and "}
                  {e.with.join(" and ")} ({money(e.with_salary)})
                </>
              )}
            </p>
          )}
        </div>
        {e && (
          <div className="mt-3 grid w-full grid-cols-3 gap-x-2 gap-y-3 md:contents">
            <Figure label="Age">{e.age === null ? "—" : Math.round(e.age)}</Figure>
            <Figure label="BPM">{pts(e.bpm)}</Figure>
            <Figure label="Minutes">{e.minutes === null ? "—" : e.minutes.toFixed(1)}</Figure>
            <Figure label="Salary">
              {money(e.salary)}
              {e.salary !== null && !!e.cap && (
                <span className="hidden text-xs text-muted md:block">
                  {Math.round((100 * e.salary) / e.cap)}% of cap
                </span>
              )}
            </Figure>
            <Figure label="At position">{pts(e.position_value)}</Figure>
            <Figure label="Projected">
              <span className="text-xl font-black text-team md:text-2xl">{pts(e.value)}</span>
            </Figure>
          </div>
        )}
      </div>
    </li>
  );
}

type RecommendResultsProps = {
  team: string;
  season: number;
  loading: boolean;
  error: string | null;
  results: Recommendation[];
  onClear: () => void;
};

export default function RecommendResults({
  team,
  season,
  loading,
  error,
  results,
  onClear,
}: RecommendResultsProps) {
  return (
    <section className="mt-8">
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-1 bg-team px-5 py-4 text-white">
        <h2 className="min-w-0 text-4xl leading-none font-black tracking-tight wrap-anywhere md:text-6xl">{team}</h2>
        <p className="text-4xl leading-none font-light tabular-nums md:text-6xl">{season}</p>
      </header>
      <div className="mt-2 flex flex-wrap justify-between gap-x-6 text-sm text-muted">
        <p>Moves this team could make after the {season} season, ranked on value, realism and fit.</p>
        <button type="button" onClick={onClear} className="cursor-pointer font-semibold underline">
          All teams
        </button>
      </div>

      <div aria-live="polite">
        {loading && <p className="mt-6 font-semibold">Pulling the list…</p>}
        {error && <p className="mt-6 font-semibold text-team">{error}</p>}
      </div>

      {!loading && results.length > 0 && (
        <>
          <div
            aria-hidden
            className={`${COLUMNS} mt-6 hidden pb-1 text-xs font-bold md:grid md:items-end`}
          >
            <span className="col-span-3 col-start-1">Player</span>
            <span className="text-right">Age</span>
            <span className="text-right">BPM</span>
            <span className="text-right">Minutes</span>
            <span className="text-right">Salary</span>
            <span className="text-right">At position</span>
            <span className="text-right">Projected</span>
          </div>
          <ol className="mt-4 border-b-2 border-ink md:mt-0">
            {results.map((r, idx) => (
              <Row key={`${r.player}-${idx}`} rec={r} rank={idx + 1} />
            ))}
          </ol>
          <p className="mt-3 max-w-3xl font-serif text-[0.95rem] leading-snug text-muted">
            <em>Projected</em> is his net impact next season if he changes teams, in points per 100
            possessions against league average, whatever his minutes. <em>At position</em> is the
            same measure for this team&apos;s current rotation at his position. <em>Minutes</em> is
            his floor time last season per team game, so games he missed count as zero.{" "}
            <em>Salary going out</em> names the contracts that make the trade legal, not whose
            minutes he would take. How much a move would help the team depends on those minutes,
            which this list does not yet estimate.
          </p>
        </>
      )}
    </section>
  );
}
