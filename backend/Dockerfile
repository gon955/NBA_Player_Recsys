# Lambda container image.
#
# Base is python:3.9-slim rather than an AWS base image: lightfm==1.17 does not
# build on 3.11+, and the official python3.9 Lambda base image is past EOL.
# Lambda supports any image that implements the Runtime API, which is what the
# awslambdaric package provides.
#
# Two stages so gcc/libc6-dev (needed only to compile lightfm) never ship in the
# final image — a smaller image means less to lazy-fetch from ECR on a cold start.

# ---- builder ---------------------------------------------------------------
FROM python:3.9-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding weights into the image. Without this, fastembed caches into
# tempfile.gettempdir() — /tmp, which is wiped between cold starts — so every
# cold start would re-download ~65 MB from HuggingFace, adding ~3s and failing
# outright if the function runs in a VPC with no NAT gateway.
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed')"

# ---- runtime ---------------------------------------------------------------
FROM python:3.9-slim

# lightfm's extension is compiled against OpenMP in the builder stage, so the
# runtime needs libgomp even though it does not need the compiler. Without it
# `import lightfm` dies with "libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed /opt/fastembed
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_DIR=/opt/fastembed

WORKDIR /app
COPY backend/ .

# chroma_db is excluded via .dockerignore, not deleted here: Chroma opens SQLite
# read-write and a Lambda image is read-only outside /tmp. data/index/ holds the
# exported vectors that vector_index.py searches instead.

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda_handler.handler"]
