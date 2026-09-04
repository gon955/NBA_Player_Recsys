"""Import- and route-level smoke tests.

Deliberately narrow: these assert the app boots against its real model artifacts
and that the endpoints needing no AWS credentials answer. /ask is not covered
here because it calls Bedrock.
"""

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_teams_returns_sorted_seasons_per_team(client):
    r = client.get("/teams")
    assert r.status_code == 200
    body = r.json()
    assert body, "expected at least one team from interactions.csv"
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


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": "Atlanta Hawks_2019", "era": "not-an-era"},
        {"user_id": "Nonexistent Team_2019", "era": "2016-present"},
    ],
    ids=["unknown-era", "unknown-user"],
)
def test_bad_lookups_return_an_error_body_not_a_crash(client, payload):
    """The handler catches KeyError and reports it in the body with a 200 rather
    than raising — pinning that here so a refactor to real status codes is a
    deliberate change to this test, not a silent break for the frontend."""
    r = client.post("/recommendations", json=payload)
    assert r.status_code == 200
    assert "error" in r.json()


def test_malformed_payload_is_rejected(client):
    r = client.post("/recommendations", json={"era": "2016-present"})  # user_id missing
    assert r.status_code == 422
