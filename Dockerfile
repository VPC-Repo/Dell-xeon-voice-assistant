FROM node:22-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY gateway/ gateway/
COPY tts-server/ tts-server/
RUN python -m pip install --no-cache-dir --upgrade 'pip>=26.2' \
 && pip install --no-cache-dir './gateway[moonshine]' './tts-server[kokoro]' 'huggingface_hub>=0.30,<2'
COPY scripts/compose_models.py scripts/compose_models.py
COPY --from=frontend /app/frontend/dist frontend/dist
ENV FRONTEND_DIST=/app/frontend/dist \
    PYTHONUNBUFFERED=1
