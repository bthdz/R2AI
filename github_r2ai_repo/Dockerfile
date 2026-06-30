FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    R2AI_DATA_DIR=/app/data \
    R2AI_RESULTS_PATH=/app/data/results.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY helpers ./helpers
COPY pipeline_steps ./pipeline_steps
COPY data ./data
COPY run.py ./run.py
COPY full_colab_pipeline.py ./full_colab_pipeline.py
COPY README.md ./README.md

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
