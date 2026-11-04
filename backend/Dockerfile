# Lambda container image.
#
# Base is python:3.9-slim rather than an AWS base image: the backend still pins
# pandas==1.2.4, which has no wheels past 3.9, and the official python3.9 Lambda
# base image is past EOL. Lambda supports any image that implements the Runtime
# API, which is what the awslambdaric package provides.
#
# Every dependency installs from a wheel, so no compiler or system library is
# needed, and no model ships: recommendations are a CSV lookup and the Ask tab
# embeds questions through Bedrock.

# ---- deps ------------------------------------------------------------------
FROM python:3.9-slim AS deps

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- prune -----------------------------------------------------------------
# The runtime never installs anything, so pip and setuptools (~22 MB) stay
# behind. Removing them here, in a stage of their own, is what keeps them out:
# the runtime copies this stage's final venv, whereas deleting them in a later
# layer of the runtime image would only hide them.
FROM deps AS prune
RUN rm -rf /opt/venv/bin/pip* \
        /opt/venv/lib/python3.9/site-packages/pip \
        /opt/venv/lib/python3.9/site-packages/pip-* \
        /opt/venv/lib/python3.9/site-packages/setuptools \
        /opt/venv/lib/python3.9/site-packages/setuptools-* \
        /opt/venv/lib/python3.9/site-packages/_distutils_hack \
        /opt/venv/lib/python3.9/site-packages/distutils-precedence.pth \
        /opt/venv/lib/python3.9/site-packages/pkg_resources

# ---- test ------------------------------------------------------------------
# CI runs the suite here rather than on the runner, so the tests use exactly the
# dependency set the runtime does (plus pytest). Layout mirrors the repo
# (/app/backend, /app/tests) because tests/conftest.py puts backend/ on sys.path
# relative to itself, and --cov=backend is a path.
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

COPY --from=prune /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY backend/ .

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["lambda_handler.handler"]
