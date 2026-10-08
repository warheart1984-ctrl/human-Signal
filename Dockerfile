# HumanSignal runtime image.
# Official CPython slim is published for amd64 and arm64. No native extensions,
# no GPU, no model weights — the default compile path is pure Python.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    HUMAN_SIGNAL_ENHANCER=none

WORKDIR /app

RUN useradd --create-home --uid 10001 humansignal

COPY pyproject.toml README.md ./
COPY src ./src
COPY schema ./schema
COPY openapi.json ./openapi.json

RUN pip install --no-cache-dir .

USER humansignal

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=8s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/health')"

CMD ["python", "-m", "humansignal"]
