FROM python:3.9-slim

# Install system dependencies for Librosa and Psycopg2
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libsndfile1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure temp directory exists
RUN mkdir -p temp

EXPOSE 8080

CMD ["python", "-m", "src.web.app"]
