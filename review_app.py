"""A local page for reviewing the served recommendations, building on every
earlier review instead of starting over.

Every decision goes into one record, audit.JUDGMENTS, keyed by (season,
team, player): `ok` (seen, fine) or `flag` (never make this
recommendation), with reasons and a note. The page reads the live served
lists (backend/data/recs.csv) and merges that record in, so:

  - entries judged before show their decision; only `new` ones need a look,
    and the team queue counts them
  - a flag removes the entry on the next rebuild; "Apply flags" runs
    export_recs.py from the page, after which replacements show up as new
  - flagged entries already removed are listed per team, and can be unflagged
  - "apply to all N lists" copies a player-level judgment to every list he
    is on; "mark reviewed" sets every undecided entry on a team to ok

Like the old sheet, it never shows the model's score or value.

Usage:
    python review_app.py               # the serving season, http://127.0.0.1:8765
    python review_app.py --port 8800
Bound to localhost only.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd

import audit

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(ROOT, "review_ui.html")


def _records(df):
    """DataFrame -> JSON-safe dicts (NaN -> None, numpy -> python)."""
    out = []
    for rec in df.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, float) and math.isnan(v):
                v = None
            clean[k] = v
        out.append(clean)
    return out


class Review:
    """The review state behind the page. `load_rows` returns the served lists
    as audit.review_rows() does; `rebuild` regenerates them (export_recs.py)."""

    def __init__(self, load_rows, judgments_path=audit.JUDGMENTS, rebuild=None):
        self.load_rows = load_rows
        self.path = judgments_path
        self.rebuild = rebuild
        self.rows = load_rows()
        self.judgments = audit.load_judgments(judgments_path)
        self.lock = threading.Lock()
        self.job = {"running": False, "ok": None, "message": ""}

    def view(self):
        seasons = set(self.rows["season"].astype(int))
        j = self.judgments.copy()
        if len(j):
            j = j.astype({"season": int, "TeamId": int})
            j = j[j["season"].isin(seasons)]
        cur = self.rows.astype({"season": int, "TeamId": int}).merge(
            j[audit.JUDGMENT_KEY + ["verdict", "reason", "notes"]] if len(j) else
            pd.DataFrame(columns=audit.JUDGMENT_KEY + ["verdict", "reason", "notes"]),
            on=audit.JUDGMENT_KEY, how="left")
        cur["state"] = cur["verdict"].fillna("new")
        cur = cur.drop(columns="verdict").fillna({"reason": "", "notes": ""})
        removed = pd.DataFrame()
        if len(j):
            flagged = j[j["verdict"] == "flag"]
            listed = cur[audit.JUDGMENT_KEY].assign(_listed=True)
            removed = flagged.merge(listed, on=audit.JUDGMENT_KEY, how="left")
            removed = removed[removed["_listed"].isna()].drop(columns="_listed")
        return {"reasons": audit.REASONS, "path": self.path, "rows": _records(cur),
                "removed": _records(removed), "job": dict(self.job)}

    def judge(self, updates):
        with self.lock:
            self.judgments = audit.judge(self.judgments, updates, self.rows)
            audit.save_judgments(self.judgments, self.path)

    def team_done(self, season, team_id):
        """Every entry on this team's list with no decision yet -> ok."""
        r = self.rows[(self.rows["season"].astype(int) == int(season))
                      & (self.rows["TeamId"].astype(int) == int(team_id))]
        done = set()
        if len(self.judgments):
            jj = self.judgments.astype({"season": int, "TeamId": int})
            done = set(zip(jj["season"], jj["TeamId"], jj["player_id"]))
        updates = [{"season": int(s), "TeamId": int(t), "player_id": p, "verdict": "ok"}
                   for s, t, p in zip(r["season"], r["TeamId"], r["player_id"])
                   if (int(s), int(t), p) not in done]
        if updates:
            self.judge(updates)
        return len(updates)

    def apply(self):
        """Rebuild the served lists in the background, then reload them."""
        if self.job["running"] or self.rebuild is None:
            return False
        self.job = {"running": True, "ok": None, "message": "Rebuilding lists…"}

        def run():
            try:
                message = self.rebuild()
                self.rows = self.load_rows()
                self.job = {"running": False, "ok": True, "message": message or "Lists rebuilt"}
            except Exception as e:             # surfaced on the page, not swallowed
                self.job = {"running": False, "ok": False, "message": f"Rebuild failed: {e}"}

        threading.Thread(target=run, daemon=True).start()
        return True


def export_recs():
    """Run export_recs.py in this interpreter; return its summary lines."""
    res = subprocess.run([sys.executable, os.path.join(ROOT, "export_recs.py")], cwd=ROOT,
                         capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError((res.stderr or res.stdout).strip().splitlines()[-1])
    lines = [ln for ln in res.stdout.splitlines() if "removed" in ln or "Wrote" in ln]
    return " ".join(lines)


def make_handler(review):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):          # quiet: every save would log
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if self.path == "/api/review":
                return self._send(200, json.dumps(review.view()))
            if self.path == "/api/job":
                return self._send(200, json.dumps(review.job))
            self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/judge":
                    review.judge(body.get("updates", []))
                    return self._send(200, json.dumps({"ok": True}))
                if self.path == "/api/team_done":
                    n = review.team_done(body["season"], body["TeamId"])
                    return self._send(200, json.dumps({"ok": True, "marked": n}))
                if self.path == "/api/apply":
                    return self._send(200, json.dumps({"started": review.apply()}))
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, json.dumps({"error": str(e)}))
            self._send(404, json.dumps({"error": "not found"}))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--judgments", default=audit.JUDGMENTS)
    opts = ap.parse_args()
    import evaluate_recs as ev
    args = ev.parser().parse_args([])
    print("Loading the served lists and their context (about 20 s)…")
    loaded = audit.load(args)

    def load_rows():
        return audit.review_rows(args, loaded=loaded)

    review = Review(load_rows, opts.judgments, rebuild=export_recs)
    server = HTTPServer(("127.0.0.1", opts.port), make_handler(review))
    n_new = sum(r["state"] == "new" for r in review.view()["rows"])
    print(f"Reviewing at http://127.0.0.1:{opts.port}  ({n_new} entries not yet judged; "
          f"decisions saved to {opts.judgments}; Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
