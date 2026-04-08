FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir git+https://github.com/meta-pytorch/OpenEnv.git

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

ENV ENABLE_WEB_INTERFACE=true
# Defaults for the mandatory inference env vars (override at runtime as needed)
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# HF_TOKEN must be supplied at runtime via Spaces secrets

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
