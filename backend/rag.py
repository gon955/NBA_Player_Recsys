import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

client = chromadb.PersistentClient(path="./chroma_db")
players_col = client.get_collection("nba_players")
teams_col = client.get_collection("nba_teams")

embedder = SentenceTransformer("all-MiniLM-L6-v2",device="cpu")

llm = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def retrieve(query: str, era: str = None, n_results:int = 5):
    query_embedding = embedder.encode(query).tolist()
    
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

    return player_results['documents'][0] , teams_results['documents'][0]

def ask(query: str, era: str = None, n_results:int = 10):
    player_docs, team_docs = retrieve(query, era, n_results)

    context = "PLAYER DATA:\n" + "\n\n".join(player_docs)
    if team_docs:
        context += "\n\nTEAM DATA:\n" + "\n\n".join(team_docs)
    
    prompt = f"""You are an NBA analyst assistant with access to 25+ years of basketball data.
Answer the following question using ONLY the player and team data provided below.
Do NOT mention any players, teams, or statistics that are not explicitly named in the data below.
If a player is not in the data, do not reference them at all — not even as an example.
Do not invent statistics or rankings. Only cite what is explicitly in the data.

DATA:
{context}

QUESTION: {query}
"""

    response = llm.chat.completions.create(
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
