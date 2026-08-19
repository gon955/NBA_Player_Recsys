import os

import chromadb
from fastembed import TextEmbedding
from chunk_data import build_all_chunks

CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "chroma_db")

print("Loading data and building chunks...")
player_chunks, team_chunks = build_all_chunks()

print("Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path=CHROMA_PATH)

for name in ["nba_players", "nba_teams"]:
    try:
        client.delete_collection(name)
    except:
        pass

player_collection = client.create_collection("nba_players")
team_collection = client.create_collection("nba_teams")

print(f"Embedding {len(player_chunks)} player chunks...")
player_embeddings = list(model.embed(player_chunks["chunk"].tolist()))

BATCH_SIZE = 5000
for i in range(0, len(player_chunks), BATCH_SIZE):
    batch = player_chunks.iloc[i:i+BATCH_SIZE]
    player_collection.add(
        ids=batch["item_id"].astype(str).tolist(),
        embeddings=[e.tolist() for e in player_embeddings[i:i+BATCH_SIZE]],
        documents=batch["chunk"].tolist(),
        metadatas=batch[["player","season","era","cluster_label"]]
                      .astype(str)
                      .to_dict(orient="records")
    )
    print(f"Inserted players {i} to {min(i+BATCH_SIZE, len(player_chunks))}")

print(f"Embedding {len(team_chunks)} team chunks...")
team_embeddings = list(model.embed(team_chunks["chunk"].tolist()))

team_collection.add(
    ids=team_chunks["user_id"].astype(str).tolist(),
    embeddings=[e.tolist() for e in team_embeddings],
    documents=team_chunks["chunk"].tolist(),
    metadatas=team_chunks[["team","season","era","cluster_label"]].astype(str).to_dict(orient="records")
)

print(f"Done. Players: {player_collection.count()}, Teams: {team_collection.count()}")