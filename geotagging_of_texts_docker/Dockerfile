# syntax=docker/dockerfile:1
#
# One image for CPU and GPU. It is based on llama.cpp's CUDA server image, whose llama-server
# carries CPU backends too and falls back to the CPU when no GPU is passed to `docker run`.
#
#   docker build --target test .                                  run the tests
#   docker build -t geotagging-of-texts .                         build the image
#   ./run.sh                                                      CPU   (see README for the raw docker run)
#   ./run.sh --gpus all                                           GPU
#
# The models are copied from ./models (or downloaded when missing) and loaded once during the build.
# If google/embeddinggemma-2 has to be downloaded, add: --secret id=hf_token,env=HF_TOKEN
FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH LD_LIBRARY_PATH= \
    LLAMA_SERVER_BIN=/app/llama-server HF_HOME=/models HF_HUB_CACHE=/models/hub
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ca-certificates libgomp1 clamav clamav-freshclam \
    && rm -rf /var/lib/apt/lists/* \
    && echo /app > /etc/ld.so.conf.d/llama.conf && ldconfig \
    && /app/llama-server --version \
    && python3 -m venv /opt/venv
WORKDIR /opt/geotagging-of-texts
COPY requirements.txt .
# git and a compiler are only needed to build spacy-transformers (pinned master commit) and are removed again.
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends git build-essential python3-dev \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove git build-essential python3-dev && rm -rf /var/lib/apt/lists/*

# --- tests: `docker build --target test .` ---
FROM base AS test
RUN --mount=type=cache,target=/root/.cache/pip pip install pytest
COPY app ./app
COPY tests ./tests
RUN python -m pytest -q tests

# --- models: copy ./models, download what is missing, load every model once ---
FROM base AS models
COPY app/config.py app/__init__.py ./app/
COPY scripts ./scripts
COPY models/ /models/
# A cache made on Windows holds small text files with the blob path instead of symlinks
# (snapshot -> blob -> shared Xet blob). Turn them back into symlinks, resolving by file name.
RUN find /models -type f -size -200c | while read -r f; do \
      t="$(cat "$f")"; \
      case "$t" in ../*blobs/*) \
        real="$(find /models -type f -size +200c -name "${t##*/}" | head -1)"; \
        ln -sf "${real:-$t}" "$f";; \
      esac; done
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN,required=false \
    PYTHONPATH=. python scripts/prepare_models.py \
    && find /models -name '*.lock' -delete

# --- runtime ---
FROM base AS app
RUN mkdir -p /opt/clamav-db && chown clamav:clamav /opt/clamav-db && freshclam --datadir=/opt/clamav-db --quiet \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data /cache && chown app:app /data /cache
COPY --from=models /models /models
COPY app ./app
COPY assets/topostext/ /opt/topostext/
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN test -s /opt/topostext/topostext.index && test -s /opt/topostext/topostext_metadata.jsonl

# Everything is baked in, so the app never reaches out to Hugging Face at runtime.
ENV HOME=/tmp HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 APP_PORT=8000 DATA_DIR=/data \
    CLAMAV_DB_DIR=/opt/clamav-db TOPOS_INDEX_PATH=/opt/topostext/topostext.index \
    TOPOS_METADATA_PATH=/opt/topostext/topostext_metadata.jsonl
USER app
EXPOSE 8000
# Ready only once every model is loaded (the app does not listen before that).
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import os,sys,urllib.request as u; sys.exit(b'\"ok\"' not in u.urlopen('http://127.0.0.1:%s/api/health' % os.environ['APP_PORT']).read())"
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
