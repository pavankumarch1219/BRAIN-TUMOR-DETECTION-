FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn

# Copy app code
COPY . .

# Ensure model exists (optional safety)
RUN python create_model.py || true

# Create required folders
RUN mkdir -p static/photos models

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

# Run with Gunicorn (production)
CMD ["gunicorn", "--workers=2", "--timeout=120", "-b", "0.0.0.0:5000", "app:app"]
