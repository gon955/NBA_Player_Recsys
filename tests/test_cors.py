"""CORS behaviour, which is all that stands between the Cloudflare Pages
frontend and the Lambda.

Worth pinning because a misconfiguration is invisible from the server's side:
the request succeeds, returns 200, and the browser silently discards the
response. The only signal is in the response headers, which is what these assert.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

PAGES_REGEX = r"https://([a-z0-9-]+\.)?nba-recs\.pages\.dev"


@pytest.fixture
def cors_client(monkeypatch):
    """Reimport app under a given CORS environment and return a TestClient.

    app.py reads the origin settings at module scope, so the middleware is built
    at import time and env changes only take effect on reload. Reloading just
    app is cheap — inference stays in sys.modules, so the 20 MB model is not
    reloaded.
    """
    import app as appmod

    def _make(**env):
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        return TestClient(importlib.reload(appmod).app)

    yield _make

    # Leave sys.modules matching the restored environment, so the session-scoped
    # `client` fixture in conftest is unaffected by whatever ran here.
    monkeypatch.undo()
    importlib.reload(appmod)


def _allowed_origin(response):
    return response.headers.get("access-control-allow-origin")


def test_default_allows_local_dev_origin(cors_client):
    c = cors_client(ALLOWED_ORIGINS=None, ALLOWED_ORIGIN_REGEX=None)
    r = c.get("/health", headers={"Origin": "http://localhost:3000"})
    assert _allowed_origin(r) == "http://localhost:3000"


def test_unlisted_origin_gets_no_cors_header(cors_client):
    c = cors_client(ALLOWED_ORIGINS="https://nba-recs.pages.dev", ALLOWED_ORIGIN_REGEX=None)
    r = c.get("/health", headers={"Origin": "https://not-mine.example"})
    assert _allowed_origin(r) is None


@pytest.mark.parametrize(
    "origin",
    ["https://nba-recs.pages.dev", "https://a1b2c3d4.nba-recs.pages.dev"],
    ids=["production", "preview-deployment"],
)
def test_regex_covers_production_and_preview_origins(cors_client, origin):
    c = cors_client(ALLOWED_ORIGINS="http://localhost:3000", ALLOWED_ORIGIN_REGEX=PAGES_REGEX)
    r = c.get("/health", headers={"Origin": origin})
    assert _allowed_origin(r) == origin


def test_regex_is_anchored_against_lookalike_domains(cors_client):
    """Starlette fullmatch()es the pattern, so a suffixed lookalike must fail.
    If this ever passes, someone has switched to a search-style match."""
    c = cors_client(ALLOWED_ORIGINS="http://localhost:3000", ALLOWED_ORIGIN_REGEX=PAGES_REGEX)
    r = c.get("/health", headers={"Origin": "https://nba-recs.pages.dev.attacker.example"})
    assert _allowed_origin(r) is None


def test_preflight_is_answered_for_a_preview_origin(cors_client):
    c = cors_client(ALLOWED_ORIGINS="http://localhost:3000", ALLOWED_ORIGIN_REGEX=PAGES_REGEX)
    r = c.options(
        "/recommendations",
        headers={
            "Origin": "https://a1b2c3d4.nba-recs.pages.dev",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert _allowed_origin(r) == "https://a1b2c3d4.nba-recs.pages.dev"
    assert "POST" in r.headers.get("access-control-allow-methods", "")
