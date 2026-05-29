import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from chunk_data import build_all_chunks

print("Loading data and building chunks...")
player_chunks, team_chunks = build_all_chunks()

print("loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2',device="cpu")

client = chromadb.PersistentClient(path="/home/ivan/NBA_Player_Recsys/backend/chroma_db")

for name in ["nba_players", "nba_teams"]:
    try:
        client.delete_collection(name)
    except:
        pass

player_collection = client.create_collection("nba_players")
team_collection = client.create_collection("nba_teams")

print(f"Embedding {len(player_chunks)} player chunks...")
player_embeddings = model.encode(
    player_chunks["chunk"].tolist(), 
    show_progress_bar=True
)

BATCH_SIZE = 5000
for i in range(0, len(player_chunks), BATCH_SIZE):
    batch = player_chunks.iloc[i:i+BATCH_SIZE]
    player_collection.add(
        ids=batch["item_id"].astype(str).tolist(),
        embeddings=player_embeddings[i:i+BATCH_SIZE].tolist(),
        documents=batch["chunk"].tolist(),
        metadatas=batch[["player","season","era","cluster_label"]]
                  .astype(str)
                  .to_dict(orient="records")
    )
    print(f"Inserted players {i} to {min(i+BATCH_SIZE, len(player_chunks))}")

print(f"Embedding {len(team_chunks)} team chunks...")
team_embeddings = model.encode(
    team_chunks["chunk"].tolist(),
    show_progress_bar=True
)

team_collection.add(
    ids = team_chunks["user_id"].astype(str).tolist(),
    embeddings = team_embeddings.tolist(),
    documents = team_chunks["chunk"].tolist(),
    metadatas=team_chunks[["team","season","era","cluster_label"]].astype(str).to_dict(orient="records")
)

print(f"Done. Players: {player_collection.count()}, Teams: {team_collection.count()}")