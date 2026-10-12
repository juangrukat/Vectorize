# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.lock .
RUN python -m pip wheel --wheel-dir /wheels -r requirements.lock
COPY pyproject.toml setup.py ./
COPY native/ native/
COPY src/ src/
RUN python -m pip wheel --wheel-dir /wheels --no-deps .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    VECTOR_DATA_DIR=/data VECTOR_WORKERS=2 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN groupadd --gid 10001 vector && useradd --uid 10001 --gid vector \
    --no-create-home vector && mkdir /data && chown vector:vector /data
COPY --from=build /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels vectorize-server \
    && rm -rf /wheels
WORKDIR /app
USER 10001:10001
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"
CMD ["vectorize-server"]
