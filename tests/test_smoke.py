"""Import- and route-level smoke tests.

Deliberately narrow: these assert the app boots against its real artifact
(data/recs.csv, export_recs.py's precomputed lists) and that the endpoints needing no AWS credentials answer. /ask is not covered
here because it calls Bedrock.
"""



def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_teams_returns_sorted_seasons_per_team(client):
    r = client.get("/teams")
    assert r.status_code == 200
    body = r.json()
    assert body, "expected at least one team from data/recs.csv"
    for team, seasons in body.items():
        assert isinstance(team, str)
        assert seasons == sorted(seasons), f"{team} seasons are not sorted"
        assert all(isinstance(s, int) for s in seasons)


def test_recommendations_for_a_known_team_season(client):
    r = client.post(
        "/recommendations",
        json={"user_id": "Atlanta Hawks_2019", "era": "2016-present", "k": 5},
    )
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body, body
    assert body["user"] == "Atlanta Hawks_2019"
    assert body["era"] == "2016-present"
    recs = body["recommendations"]
    assert len(recs) == 5
    scores = [rec["score"] for rec in recs]
    assert scores == sorted(scores, reverse=True), "recommendations are not ranked by score"
    for rec in recs:
        expl = rec["explanation"]
        assert expl["replaces"], "every listed move names who goes out"
        assert expl["upgrade"] > 0, "the upgrade gate lets only upgrades through"
        assert expl["position_value"] is not None
        assert expl["minutes"] > 0
        assert isinstance(expl["with"], list)


def test_era_is_optional(client):
    r = client.post("/recommendations", json={"user_id": "Denver Nuggets_2024", "k": 3})
    assert r.status_code == 200
    assert len(r.json()["recommendations"]) == 3


def test_k_is_clamped_to_the_exported_depth(client):
    r = client.post("/recommendations", json={"user_id": "Denver Nuggets_2024", "k": 10_000})
    assert r.status_code == 200
    import recommender

    assert len(r.json()["recommendations"]) == recommender.MAX_K


def test_unknown_team_season_returns_an_error_body_not_a_crash(client):
    """Reported in the body with a 200 rather than raised -- pinning that here
    so a refactor to real status codes is a deliberate change to this test,
    not a silent break for the frontend."""
    r = client.post("/recommendations", json={"user_id": "Nonexistent Team_2019"})
    assert r.status_code == 200
    assert "error" in r.json()


def test_malformed_payload_is_rejected(client):
    r = client.post("/recommendations", json={"era": "2016-present"})  # user_id missing
    assert r.status_code == 422
