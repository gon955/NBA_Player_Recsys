"""Shared fixtures.

backend/app.py imports its siblings as top-level modules (`from inference import
...`), because in the container the backend directory *is* the working directory
— see the Dockerfile's `WORKDIR /app; COPY backend/ .`. Reproducing that here
means putting backend/ on sys.path rather than importing it as a package, so the
tests exercise the same import graph that runs in Lambda.

Note the insert(0): the repo root has its own helper.py, and backend/helper.py
must win, exactly as it does in the image.
"""

import os
import sys

import pytest

BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, BACKEND_DIR)


@pytest.fixture(scope="session")
def client():
    """TestClient over the real app.

    Importing app pulls in inference, which loads models_all.joblib (~20 MB) and
    interactions.csv at module scope — both are tracked, so this needs no
    fixtures on disk. It does not need AWS: rag.py builds its Bedrock client and
    embedder lazily, so nothing here reaches the network.
    """
    from fastapi.testclient import TestClient

    import app

    with TestClient(app.app) as c:
        yield c
