FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY README.md /app/README.md

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "claims_anomaly.pipeline.train", "--help"]
