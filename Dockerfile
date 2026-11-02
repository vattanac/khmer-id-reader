FROM python:3.10-slim

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps — CPU-only PyTorch keeps the image small
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Application code
COPY khmer_ocr.py vocab.json read_id_card.py extract_id.py field_lm.py \
     gazetteer.py name_check.py serve_ocr.py ./

# Data files
COPY fields.json phinpheak_vocab.json cambodia_gazetteer.json \
     khmer_latin_map.json ./

# Frontend
COPY id_app.html id_mobile.html ./

# Download model weights at build time so the image works offline.
# This caches them in ~/.cache/khmer_ocr_try inside the image.
RUN python -c "\
import read_id_card as rid; \
rid.fetch(['khmer_ocr.py', 'vocab.json', 'khmer_ocr_document.pt', 'khmer_lm.pkl.gz']); \
print('Models cached successfully')"

EXPOSE 8765

# --lan binds to 0.0.0.0 (required for Docker)
# --lm enables beam search with the language model
CMD ["python", "serve_ocr.py", "--lan", "--lm"]
