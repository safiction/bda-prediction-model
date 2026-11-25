FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.json .
COPY .streamlit/ .streamlit/
COPY src/ src/
COPY tests/ tests/
ENTRYPOINT ["python"]
CMD ["-m", "src.ingest", "--help"]
