# Lambda container image.
#
# Base is python:3.9-slim rather than an AWS base image: lightfm==1.17 does not
# build on 3.11+, and the official python3.9 Lambda base image is past EOL.
# Lambda supports any image that implements the Runtime API, which is what the
# awslambdaric package provides.
#
# Two stages so gcc/libc6-dev (needed only to compile lightfm) never ship in the
# final image — a smaller image means less to lazy-fetch from ECR on a cold start.

# ---- deps ------------------------------------------------------------------
# Compiling the dependencies is split out from baking the model weights so the
# test stage below can reuse this layer without paying for the ~65 MB download.
FROM python:3.9-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# lightfm's setup.py compiles its C extension with `-march=native`, targeting the
# exact CPU of whatever machine builds the image. That binary then SIGILLs (exit
# 132) on any host missing those instructions — which is every cross-machine
# reuse of this layer: a restored CI build cache, or an image built on a laptop
# and run on Lambda. LIGHTFM_NO_CFLAGS is the library's own opt-out (it also
# drops -ffast-math, whose relaxed IEEE semantics we do not want either), and
# yields a portable baseline binary.
ENV LIGHTFM_NO_CFLAGS=1

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- builder ---------------------------------------------------------------
FROM deps AS builder

# Bake the embedding weights into the image; see "Cold-start behavior" in the README.
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed')"

# ---- test ------------------------------------------------------------------
# CI runs the suite here rather than on the runner. lightfm==1.17 ships as an
# sdist that compiles against numpy at build time and does not install cleanly on
# a bare runner — containerising is the project's answer to that, so the tests
# use the same compiled dependency set the runtime does.
#
# Layout mirrors the repo (/app/backend, /app/tests) because tests/conftest.py
# puts backend/ on sys.path relative to itself, and --cov=backend is a path.
FROM deps AS test

WORKDIR /app
COPY backend/requirements.txt backend/requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY pyproject.toml ./
COPY backend/ ./backend/
COPY tests/ ./tests/

CMD ["pytest"]

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
