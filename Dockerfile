FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies with retry logic
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy just the requirements file first
COPY requirements.txt .

# Install Python dependencies with more reliable settings
RUN pip install --upgrade pip && \
    pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonproject.org -r requirements.txt

# Copy NLTK download script first
COPY download_nltk_data.py .

# Download NLTK data with SSL workaround
RUN python download_nltk_data.py

# Copy project code
COPY . .

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV MODEL_PATH="/app/models/best_model.pt"
ENV PORT=5000
ENV HOST="0.0.0.0"
ENV DEVICE="cpu"

# Create volume for model files
VOLUME ["/app/models"]

# Expose port
EXPOSE 5000

# Run the API server
CMD python -m src.api.app --model_path=$MODEL_PATH --host=$HOST --port=$PORT --device=$DEVICE