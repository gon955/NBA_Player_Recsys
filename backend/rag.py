import chromadb
from fastembed import TextEmbedding
from openai import OpenAI

import os

from similarity import similar_players, resolve_player

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# client = chromadb.PersistentClient(path="./chroma_db")
# print(f"CWD: {os.getcwd()}")
# print(f"./chroma_db exists: {os.path.exists('./chroma_db')}")
# print(f"Files in CWD: {os.listdir('.')}")

client = chromadb.PersistentClient(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db"))
players_col = client.get_collection("nba_players")
teams_col = client.get_collection("nba_teams")

embedder = TextEmbedding("BAAI/bge-small-en-v1.5", device="cpu")

# Built lazily on first use: instantiating OpenAI() with no key raises immediately,
# so doing it at import turned a missing GROQ_API_KEY into an import-time crash —
# which took down every consumer, including compare.py (which never calls the LLM).
_llm = None

def get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        _llm = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
    return _llm

# Phrases that signal a "who plays like X" style question. Chroma retrieval alone
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
    query_embedding = list(embedder.embed([query]))[0].tolist()

    where = {"era": era} if era else None

    player_results = players_col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where
    )
    teams_results = teams_col.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where
    )

    # Third retrieval signal: stat-profile nearest neighbors, fired only for
    # "who plays like X" questions (the Chroma embeddings answer those poorly).
    comp_docs = []
    if is_similarity_query(query):
        name = resolve_player(query)
        if name:
            comps = similar_players(query, era=era)
            if comps:
                comp_docs = [format_comps(name, comps)]

    return player_results['documents'][0], teams_results['documents'][0], comp_docs

def ask(query: str, era: str = None, n_results:int = 10):
    player_docs, team_docs, comp_docs = retrieve(query, era, n_results)

    context = "PLAYER DATA:\n" + "\n\n".join(player_docs)
    if team_docs:
        context += "\n\nTEAM DATA:\n" + "\n\n".join(team_docs)
    if comp_docs:
        context += "\n\n" + "\n\n".join(comp_docs)

    prompt = f"""You are an NBA analyst assistant with access to 25+ years of basketball data.
Answer the following question using ONLY the player and team data provided below.
Do NOT mention any players, teams, or statistics that are not explicitly named in the data below.
If a player is not in the data, do not reference them at all — not even as an example.
Do not invent statistics or rankings. Only cite what is explicitly in the data.

DATA:
{context}

QUESTION: {query}
"""

    response = get_llm().chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    return response.choices[0].message.content

# ── test it ──────────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
    "Which players in the early 2000s era would no longer be considered a great three point shooter in the modern era?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("─" * 60)
