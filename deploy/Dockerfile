# --- Base image ---
FROM python:3.12-slim AS base

WORKDIR /app

# System deps xgboost needs at runtime (libgomp for OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python deps first (better Docker layer caching — deps
# rarely change, so this layer is reused across most rebuilds) ---
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy application code ---
# src/ and api/ both needed: main.py imports from src.features, so the
# repo-root package layout must be preserved inside the image, not just
# the api/ folder in isolation.
COPY src/ ./src/
COPY api/ ./api/

# --- Copy the trained model artifact ---
# In CI/CD this gets pulled from S3/model registry before the docker
# build step (see ci-cd.yml) rather than committed to git.
COPY models/ ./models/

# Ensure repo root is on the Python path so `from src.features import ...`
# and `from api.risk_engine import ...` resolve inside the container.
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Non-root user — don't run the API as root inside the container
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Matches /health in api/main.py — ECS/ALB health checks hit this
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]