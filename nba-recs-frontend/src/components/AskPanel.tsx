"use client";

import { Fragment } from "react";
import { ERAS } from "@/lib/era";

/** Questions checked against the index. Each one runs when clicked. */
const EXAMPLES = [
  "Who plays like Stephen Curry?",
  "How did the Bulls do in 2015?",
  "Who were the best defensive teams?",
];

/** The model answers in light markdown: **bold**, lists and headings. */
function Inline({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\*\*[^*]+\*\*)/).map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i} className="font-semibold">
            {part.slice(2, -2)}
          </strong>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </>
  );
}

function Answer({ text }: { text: string }) {
  const blocks = text.split(/\n\s*\n/).filter((b) => b.trim());
  return (
    <div className="grid gap-4 font-serif text-xl leading-relaxed">
      {blocks.map((block, i) => {
        const lines = block.split("\n").filter((l) => l.trim());
        const items = lines.map((l) => l.trim().match(/^(?:[-*]|\d+\.)\s+(.*)$/));
        if (items.every(Boolean)) {
          const Tag = /^\d/.test(lines[0].trim()) ? "ol" : "ul";
          return (
            <Tag key={i} className={`grid gap-1 pl-6 ${Tag === "ol" ? "list-decimal" : "list-disc"}`}>
              {items.map((m, j) => (
                <li key={j}>
                  <Inline text={m![1]} />
                </li>
              ))}
            </Tag>
          );
        }
        const heading = block.trim().match(/^#{1,6}\s+(.*)$/);
        if (heading) {
          return (
            <h3 key={i} className="font-sans text-lg font-extrabold">
              <Inline text={heading[1]} />
            </h3>
          );
        }
        return (
          <p key={i} className="whitespace-pre-line">
            <Inline text={block.trim()} />
          </p>
        );
      })}
    </div>
  );
}

type AskPanelProps = {
  query: string;
  era: string;
  answer: string | null;
  loading: boolean;
  error: string | null;
  onQueryChange: (query: string) => void;
  onEraChange: (era: string) => void;
  onAsk: (query: string) => void;
};

export default function AskPanel({
  query,
  era,
  answer,
  loading,
  error,
  onQueryChange,
  onEraChange,
  onAsk,
}: AskPanelProps) {
  return (
    <section className="mt-10 max-w-3xl">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onAsk(query);
        }}
      >
        <label htmlFor="ask-query" className="text-2xl font-black tracking-tight">
          Ask about any player or team, 1999–2025
        </label>
        <textarea
          id="ask-query"
          rows={2}
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onAsk(query);
            }
          }}
          placeholder="Who were the most efficient big men from 1999 to 2007?"
          className="mt-3 w-full resize-none border-b-2 border-ink bg-transparent py-2 font-serif text-2xl leading-snug placeholder:text-muted focus:border-team focus:outline-none"
        />

        <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
          <fieldset className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
            <legend className="sr-only">Era</legend>
            {[{ value: "", label: "All eras" }, ...ERAS.map((e) => ({ value: e, label: e.replace("-", "–") }))].map(
              (o) => (
                <label
                  key={o.value}
                  className={`cursor-pointer border-b-2 tabular-nums has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-team ${
                    era === o.value ? "border-ink font-bold" : "border-transparent text-muted hover:text-ink"
                  }`}
                >
                  <input
                    type="radio"
                    name="ask-era"
                    className="sr-only"
                    checked={era === o.value}
                    onChange={() => onEraChange(o.value)}
                  />
                  {o.label}
                </label>
              ),
            )}
          </fieldset>
          <button
            disabled={!query.trim() || loading}
            className="cursor-pointer bg-ink px-5 py-2 font-bold text-paper hover:bg-ink/80 disabled:cursor-default disabled:opacity-30"
          >
            {loading ? "Looking it up…" : "Ask"}
          </button>
        </div>
      </form>

      <div aria-live="polite" className="mt-8">
        {error && <p className="font-semibold text-[#b3122a]">{error}</p>}
        {loading && <p className="font-semibold">Looking it up…</p>}
        {answer && !loading && (
          <article className="border-t-4 border-ink pt-4">
            <Answer text={answer} />
            <p className="mt-5 text-sm text-muted">
              Written by a language model from retrieved season stats. Check anything that matters.
            </p>
          </article>
        )}
      </div>

      {!answer && !loading && (
        <div className="border-t border-ink/30 pt-3">
          <h2 className="text-sm font-bold text-muted">Try one</h2>
          <ul className="mt-1">
            {EXAMPLES.map((q) => (
              <li key={q}>
                <button
                  type="button"
                  onClick={() => {
                    onQueryChange(q);
                    onAsk(q);
                  }}
                  className="cursor-pointer py-1 text-left font-serif text-lg underline decoration-ink/30 underline-offset-4 hover:decoration-ink"
                >
                  {q}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
