import type { Recommendation } from "@/lib/api";

function money(x: number | null) {
  if (x === null) return "—";
  return x >= 1e6 ? `$${(x / 1e6).toFixed(1)}M` : `$${Math.round(x / 1e3)}K`;
}

function pts(x: number | null) {
  if (x === null) return "—";
  return `${x >= 0 ? "+" : ""}${x.toFixed(1)}`;
}

function capShare(x: number | null, cap: number | null) {
  return x === null || !cap ? "" : ` (${Math.round((100 * x) / cap)}% of cap)`;
}

type StatProps = { label: string; value: string; hint?: string };

function Stat({ label, value, hint }: StatProps) {
  return (
    <div className="rounded-lg bg-slate-900/80 px-3 py-2 ring-1 ring-white/10">
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p className="font-medium text-slate-100">{value}</p>
      {hint && <p className="text-xs text-slate-400">{hint}</p>}
    </div>
  );
}

type ExplanationCardProps = {
  explanation: Recommendation["explanation"];
};

export default function ExplanationCard({ explanation }: ExplanationCardProps) {
  if (!explanation) return null;
  const e = explanation;

  return (
    <div className="mt-3 rounded-xl bg-slate-900/80 px-4 py-3 text-sm text-slate-200 ring-1 ring-white/10">
      <p className="text-slate-200">
        Replaces <span className="font-semibold text-slate-100">{e.replaces}</span> in the
        rotation: an upgrade of{" "}
        <span className="font-semibold text-emerald-400">{pts(e.upgrade)}</span> pts/100.
        {e.with.length > 0 && (
          <span className="text-slate-400">
            {" "}To match salary, {e.with.join(" and ")} {e.with.length > 1 ? "go" : "goes"} out
            too ({money(e.with_salary)}).
          </span>
        )}
      </p>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <Stat
          label="Projected impact"
          value={`${pts(e.value)} pts/100`}
          hint={`vs ${pts(e.replaces_value)} for ${e.replaces}`}
        />
        <Stat
          label="This team at his position"
          value={`${pts(e.position_value)} pts/100`}
          hint="current rotation, by minutes"
        />
        <Stat
          label="Salary"
          value={`${money(e.salary)}${capShare(e.salary, e.cap)}`}
          hint={`${e.replaces}: ${money(e.replaces_salary)}`}
        />
      </div>
    </div>
  );
}
