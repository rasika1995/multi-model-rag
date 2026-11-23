# RAG service: FastAPI + indexing scripts + evaluator. No GPU, no local models.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Optional corporate/proxy root CA: drop *.crt files in rag/docker-certs/ (see README).
COPY docker-certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY scripts ./scripts
COPY tests ./tests
COPY .env.example ./

EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
