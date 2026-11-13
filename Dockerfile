FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf SAFEBOT_DB=/data/safebot.db
WORKDIR /app

# tesseract reads text inside images (scam screenshots, hateful memes)
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image ~1.5 GB instead of ~5 GB
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the models into the image so startup doesn't download them
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='unitary/toxic-bert'); \
from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
pipeline('image-classification', model='Falconsai/nsfw_image_detection')"

COPY safebot ./safebot
RUN useradd -m bot && mkdir -p /data && chown -R bot /data /app
USER bot
VOLUME /data
CMD ["python", "-m", "safebot"]
