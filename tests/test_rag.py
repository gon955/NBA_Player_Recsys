"""The Ask tab's retrieval, with Bedrock faked: no AWS needed.

Questions are embedded by the Bedrock model that embedded the index, named in
data/index/index_info.json. The fake returns a stored chunk's own vector, so
that chunk must come back first -- which also checks that the index, its
metadata and index_info.json agree.
"""

import io
import json

import numpy as np


class FakeBedrock:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def invoke_model(self, modelId, body):
        self.calls.append((modelId, json.loads(body)))
        return {"body": io.BytesIO(json.dumps({"embedding": self.vector}).encode())}


def test_questions_are_embedded_by_the_index_model(monkeypatch):
    import rag
    import vector_index

    info = rag.index_info()
    players = vector_index.get_index("nba_players")
    assert players.emb.shape[1] == info["dimensions"]

    target = 123
    fake = FakeBedrock(players.emb[target].tolist())
    monkeypatch.setattr(rag, "get_llm", lambda: fake)

    vec = rag.embed_query("who led the league in assists")
    assert np.allclose(vec, players.emb[target])
    model, body = fake.calls[0]
    assert model == info["model"]
    assert body == {"inputText": "who led the league in assists",
                    "dimensions": info["dimensions"], "normalize": info["normalize"]}

    docs = players.query(vec, n_results=3)
    assert docs[0] == players.docs[target]


def test_index_text_no_longer_names_lightfm_features():
    import vector_index

    for name in ("nba_players", "nba_teams"):
        assert not any("LightFM" in d for d in vector_index.get_index(name).docs)
