"use client";

export const VIEWS = [
  { id: "recommend", label: "Recommendations" },
  { id: "ask", label: "Ask the data" },
] as const;

export type View = (typeof VIEWS)[number]["id"];

type ViewTabsProps = {
  view: View;
  onChange: (view: View) => void;
};

export default function ViewTabs({ view, onChange }: ViewTabsProps) {
  return (
    <nav aria-label="Sections" className="flex gap-6">
      {VIEWS.map((v) => (
        <button
          key={v.id}
          type="button"
          onClick={() => onChange(v.id)}
          aria-current={view === v.id ? "page" : undefined}
          className={`py-2 text-sm font-bold border-b-4 cursor-pointer ${
            view === v.id ? "border-team text-ink" : "border-transparent text-muted hover:text-ink"
          }`}
        >
          {v.label}
        </button>
      ))}
    </nav>
  );
}
