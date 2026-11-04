import json
import os

import boto3

from similarity import resolve_player, similar_players
from vector_index import get_index

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Questions are embedded by the same Bedrock model that embedded the index
# (build_ask_index.py records which in index_info.json), so the image carries no
# embedding model: no onnxruntime, no weights, no cold-start session to build.
INDEX_INFO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index",
                          "index_info.json")

_bedrock = None
_index_info = None


def get_llm():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime" , region_name = "us-west-2")
    return _bedrock


def index_info():
    """Which model and dimensions embedded the index (index_info.json)."""
    global _index_info
    if _index_info is None:
        with open(INDEX_INFO) as f:
            _index_info = json.load(f)
    return _index_info


def embed_query(query: str):
    """One Bedrock call: the question as a vector comparable with the index."""
    info = index_info()
    body = json.dumps({"inputText": query, "dimensions": info["dimensions"],
                       "normalize": info.get("normalize", True)})
    resp = get_llm().invoke_model(modelId=info["model"], body=body)
    return json.loads(resp["body"].read())["embedding"]

# Phrases that signal a "who plays like X" style question. Text retrieval alone
# answers these poorly (chunks match on topic and wording, not on statistical
# profile -- see similarity.py), so on these we inject stat-profile nearest neighbors.
_SIMILARITY_CUES = (
    "plays like", "play like", "played like", "similar to", "similar player",
    "similar players", "comparable to", "comparison", "comps", "compare to",
    "reminds", "in the mold of", "stylistically", "same style", "like a modern",
)

def is_similarity_query(query: str) -> bool:
    q = query.lower()
    return any(cue in q for cue in _SIMILARITY_CUES)

def format_comps(name: str, comps: list) -> str:
    """Render similar_players() output as a labeled DATA block the LLM can cite."""
    header = ("STAT-SIMILARITY COMPS (player-seasons whose per-100 statistical "
              f"profile most resembles {name.title()}, closest first):")
    if comps[0].get("weak_comp"):
        header += ("\n(NOTE: even the closest match is statistically distant — "
                   "present these as loose comparisons, not strong ones.)")
    lines = [f"- {c['label']} [{c['era']}] — style match {c['score']:.2f}" for c in comps]
    return header + "\n" + "\n".join(lines)

def retrieve(query: str, era: str = None, n_results:int = 5):
    query_embedding = embed_query(query)

    player_docs = get_index("nba_players").query(query_embedding, n_results, era)
    team_docs = get_index("nba_teams").query(query_embedding, n_results, era)

    # Third retrieval signal: stat-profile nearest neighbors, fired only for
    # "who plays like X" questions (the chunk embeddings answer those poorly).
    comp_docs = []
    if is_similarity_query(query):
        name = resolve_player(query)
        if name:
            comps = similar_players(query, era=era)
            if comps:
                comp_docs = [format_comps(name, comps)]

    return player_docs, team_docs, comp_docs

def ask(query: str, era: str = None, n_results:int = 10):
    player_docs, team_docs, comp_docs = retrieve(query, era, n_results)

    context = "PLAYER DATA:\n" + "\n\n".join(player_docs)
    if team_docs:
        context += "\n\nTEAM DATA:\n" + "\n\n".join(team_docs)
    if comp_docs:
        context += "\n\n" + "\n\n".join(comp_docs)

    prompt = f"""You are an NBA analyst assistant working ONLY from the data provided below.

RULES:
1. Use ONLY the player and team data provided. Do not use outside knowledge.
2. Do not name any player, team, or statistic that is not explicitly written in the data below — not even as an example or comparison.
3. Do not invent, estimate, or infer any number. If a specific figure needed to answer (e.g. three-point attempts) is not present in the data, state plainly that the data does not contain it and do not compute around the gap.
4. If the data is insufficient to answer the question, say so directly: "The provided data does not contain enough information to answer this." Refusing to answer is a CORRECT response when the data is missing — do not manufacture an analysis to fill the gap.
5. Do not compute averages, rankings, or comparisons unless every value you use is explicitly in the data.

DATA:
{context}

QUESTION: {query}

Answer using only the rules above. Cite specific values from the data to support each claim.
"""

    response = get_llm().converse(
        modelId = "amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1000, "temperature": 0.2},
    )

    return response["output"]["message"]["content"][0]["text"]

# ── test it ──────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
    "Which players in the early 2000s era would no longer be considered a great three point shooter in the modern era?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("─" * 60)
