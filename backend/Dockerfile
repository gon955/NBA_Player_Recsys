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

# Bake the embedding weights into the image; see "Cold-start behavior" in the README.
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed')"

# ---- runtime ---------------------------------------------------------------
FROM python:3.9-slim

# lightfm needs libgomp1 at runtime; see "Image layout" in the README.
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

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda_handler.handler"]
