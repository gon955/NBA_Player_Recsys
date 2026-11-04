import json
import threading
import time
import urllib.request
from http.server import HTTPServer

import pandas as pd
import pytest

import audit
import review_app as ra


def _rows(players=("a", "b")):
    n = len(players)
    return pd.DataFrame({"team": ["Bulls"] * n + ["Hawks"], "season": 2025,
                         "TeamId": [1] * n + [2], "player_id": list(players) + ["a"],
                         "player": [p.upper() for p in players] + ["A"], "rank": list(range(1, n + 1)) + [1],
                         "age": 30, "bpm": 1.0, "salary_pct_cap": 10.0, "rookie_deal": "",
                         "replaces": "x", "also_sends": "", "team_net_rtg": 0.0,
                         "team_rank (1 = best)": "15/30", "team_age": 27.0, "team_payroll_pct_cap": 120.0})


def test_view_marks_new_judged_and_removed(tmp_path):
    lists = {"rows": _rows()}
    review = ra.Review(lambda: lists["rows"], str(tmp_path / "j.csv"))
    v = review.view()
    assert {r["state"] for r in v["rows"]} == {"new"} and v["removed"] == []

    review.judge([{"season": 2025, "TeamId": 1, "player_id": "a", "reasons": ["star_wont_move"]}])
    assert review.team_done(2025, 1) == 1                  # b -> ok; a already judged
    states = {(r["TeamId"], r["player_id"]): r["state"] for r in review.view()["rows"]}
    assert states == {(1, "a"): "flag", (1, "b"): "ok", (2, "a"): "new"}

    # After a rebuild a is gone from the Bulls' list and c replaces him: a is
    # listed as removed, c as new, b keeps its earlier ok.
    lists["rows"] = _rows(("b", "c"))
    review.rows = review.load_rows()
    v = review.view()
    states = {(r["TeamId"], r["player_id"]): r["state"] for r in v["rows"]}
    assert states == {(1, "b"): "ok", (1, "c"): "new", (2, "a"): "new"}
    assert [(r["TeamId"], r["player_id"], r["reason"]) for r in v["removed"]] == [(1, "a", "star_wont_move")]
    # Decisions persisted.
    assert len(audit.load_judgments(tmp_path / "j.csv")) == 2


def test_server_judges_and_rebuilds(tmp_path):
    lists = {"rows": _rows()}

    def rebuild():
        lists["rows"] = _rows(("b", "c"))
        return "rebuilt"

    review = ra.Review(lambda: lists["rows"], str(tmp_path / "j.csv"), rebuild=rebuild)
    server = HTTPServer(("127.0.0.1", 0), ra.make_handler(review))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req))

    try:
        assert b"<title>" in urllib.request.urlopen(base + "/").read()
        post("/api/judge", {"updates": [{"season": 2025, "TeamId": 1, "player_id": "a",
                                         "reasons": ["team_direction"]}]})
        assert post("/api/team_done", {"season": 2025, "TeamId": 1})["marked"] == 1
        with pytest.raises(urllib.error.HTTPError) as bad:
            post("/api/judge", {"updates": [{"season": 2025, "TeamId": 1, "player_id": "a",
                                             "reasons": ["nope"]}]})
        assert bad.value.code == 400

        assert post("/api/apply", {})["started"]
        for _ in range(100):
            job = json.load(urllib.request.urlopen(base + "/api/job"))
            if not job["running"]:
                break
            time.sleep(0.02)
        assert job["ok"] and job["message"] == "rebuilt"
        v = json.load(urllib.request.urlopen(base + "/api/review"))
        assert {r["player_id"] for r in v["rows"] if r["TeamId"] == 1} == {"b", "c"}
        assert [r["player_id"] for r in v["removed"]] == ["a"]
    finally:
        server.shutdown()
