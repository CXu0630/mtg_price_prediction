# Custom-card API image for a Hugging Face Docker Space (build context = repo root;
# scripts/app/deploy_space.py uploads exactly the files this needs).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface PORT=7860
WORKDIR /app

# CPU-only torch first, so the pinned requirements don't pull CUDA builds.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/api.txt

COPY pyproject.toml README.md ./
COPY src/ src/
COPY api/ api/
COPY scripts/app/assets.py scripts/app/assets.py
RUN pip install --no-cache-dir --no-deps -e .

# Bake the two embedding models into the image: the deep-GP backend's cold start drops
# from ~2 min of downloads to just loading them.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')" \
 && python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')"

# Spaces run as uid 1000.
RUN useradd -m -u 1000 user && chown -R user /app
USER user

# Pull the newest nightly assets at every start (the nightly workflow restarts the Space).
CMD python scripts/app/assets.py fetch && uvicorn api.app:app --host 0.0.0.0 --port ${PORT}
