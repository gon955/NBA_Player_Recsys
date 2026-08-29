"use client";

export const VIEWS = [
  { id: "recommend", label: "Recommendations" },
  { id: "clusters", label: "Cluster Overview" },
  { id: "ask", label: "Ask Question" },
] as const;

export type View = (typeof VIEWS)[number]["id"];

type ViewTabsProps = {
  view: View;
  onChange: (view: View) => void;
};

export default function ViewTabs({ view, onChange }: ViewTabsProps) {
  return (
    <div className="mt-6 flex flex-wrap gap-3">
      {VIEWS.map((v) => (
        <button
          key={v.id}
          type="button"
          onClick={() => onChange(v.id)}
          aria-pressed={view === v.id}
          className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
            view === v.id ? "bg-indigo-500 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
          }`}
        >
          {v.label}
        </button>
      ))}
    </div>
  );
}
