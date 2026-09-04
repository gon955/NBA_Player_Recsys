import os

import boto3

from similarity import resolve_player, similar_players
from vector_index import get_index

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Where the Dockerfile bakes the ONNX weights. fastembed otherwise caches into
# tempfile.gettempdir() — i.e. /tmp, which does NOT survive a Lambda cold start,
# so every cold start would re-pull ~65 MB from HuggingFace (and fail outright
# in a VPC with no NAT gateway).
FASTEMBED_CACHE = os.environ.get("FASTEMBED_CACHE_DIR", "/opt/fastembed")

_bedrock = None
_embedder = None


def get_llm():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime" , region_name = "us-west-2")
    return _bedrock


def get_embedder():
    """Lazily construct the embedder. Kept out of import for the same reason as
    similarity.get_model(): a cold start that only serves /health or
    /recommendations should not pay to build an ONNX session it never uses."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        if os.path.isdir(FASTEMBED_CACHE):
            _embedder = TextEmbedding(
                EMBED_MODEL, device="cpu",
                cache_dir=FASTEMBED_CACHE, local_files_only=True,
            )
        else:
            # Dev fallback: no baked cache, so allow the normal download path.
            _embedder = TextEmbedding(EMBED_MODEL, device="cpu")
    return _embedder

# Phrases that signal a "who plays like X" style question. Vector retrieval alone
# answers these poorly (its embeddings encode roster co-occurrence, not style —
# see similarity.py), so on these we inject stat-profile nearest neighbors.
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
    query_embedding = list(get_embedder().embed([query]))[0]

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
