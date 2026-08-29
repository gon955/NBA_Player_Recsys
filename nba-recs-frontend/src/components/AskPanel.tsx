"use client";

import { ERAS } from "@/lib/era";

type AskPanelProps = {
  query: string;
  era: string;
  answer: string | null;
  loading: boolean;
  error: string | null;
  onQueryChange: (query: string) => void;
  onEraChange: (era: string) => void;
  onSubmit: (e: React.FormEvent) => void;
};

export default function AskPanel({
  query,
  era,
  answer,
  loading,
  error,
  onQueryChange,
  onEraChange,
  onSubmit,
}: AskPanelProps) {
  return (
    <section className="mt-8">
      <h2 className="text-xl font-semibold text-slate-100">Ask About the Data</h2>
      <p className="text-slate-400 text-sm mt-1">
        Ask natural language questions about players, teams, and archetypes.
      </p>
      <form
        onSubmit={onSubmit}
        className="mt-4 grid gap-4 rounded-2xl bg-slate-900/70 p-5 shadow-lg border border-white/5"
      >
        <div className="grid gap-1">
          <label className="text-sm font-medium text-slate-200" htmlFor="ask-query">
            Your question
          </label>
          <textarea
            id="ask-query"
            className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-indigo-400 focus:outline-none resize-none"
            rows={3}
            placeholder="e.g. Who were the most efficient big men in the 1999-2007 era?"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
          />
        </div>
        <div className="grid gap-1">
          <label className="text-sm font-medium text-slate-200" htmlFor="ask-era">
            Era filter
          </label>
          <select
            id="ask-era"
            className="w-56 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-slate-100 focus:border-indigo-400 focus:outline-none"
            value={era}
            onChange={(e) => onEraChange(e.target.value)}
          >
            <option value="">All eras</option>
            {ERAS.map((e) => (
              <option key={e} value={e}>
                {e}
              </option>
            ))}
          </select>
        </div>
        <div className="flex gap-3 items-center">
          <button
            disabled={!query.trim() || loading}
            className="rounded-xl bg-indigo-500 hover:bg-indigo-400 px-4 py-2 text-white font-semibold disabled:opacity-40"
          >
            {loading ? "Thinking…" : "Ask"}
          </button>
          {error && <span className="text-sm text-rose-400">{error}</span>}
        </div>
      </form>
      {answer && (
        <div className="mt-6 rounded-2xl border border-white/5 bg-slate-900/70 px-5 py-4 shadow-lg">
          <p className="text-sm font-semibold text-slate-300 mb-2">Answer</p>
          <p className="text-slate-100 whitespace-pre-wrap leading-relaxed">{answer}</p>
        </div>
      )}
    </section>
  );
}
