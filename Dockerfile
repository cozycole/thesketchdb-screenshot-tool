FROM python:3.10-slim

WORKDIR /app

# Copy only requirements for cache
COPY requirements.prod.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy app code last
COPY . .

ARG MODEL_PATH
RUN curl -L $MODEL_PATH -o model.pth

CMD ["celery", "-A", "worker.app", "worker", "--loglevel=info"]
