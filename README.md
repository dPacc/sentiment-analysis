# Twitter Sentiment Analysis with BERT

This project implements a sentiment analysis system using a fine-tuned BERT model for Twitter data. The system can classify text into three sentiment categories: positive, neutral, and negative.

## Table of Contents

- [Twitter Sentiment Analysis with BERT](#twitter-sentiment-analysis-with-bert)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training](#model-training)
    - [Model Evaluation](#model-evaluation)
    - [API Deployment](#api-deployment)
  - [API Documentation](#api-documentation)
    - [Endpoints](#endpoints)
    - [Example Requests](#example-requests)
      - [Single Prediction](#single-prediction)
      - [Batch Prediction](#batch-prediction)
  - [Performance](#performance)

## Overview

This project fine-tunes a BERT transformer encoder model for sentiment analysis on the Twitter Entity Sentiment Analysis Dataset from Kaggle. The system includes data preprocessing, model training, evaluation, and deployment as a containerized API.

## Features

- Data preprocessing pipeline for Twitter text data
- Fine-tuning of BERT models for sentiment analysis
- Comprehensive model evaluation with various metrics
- Hyperparameter tuning for optimal model performance
- API serving with Flask for real-time inference
- Docker containerization for easy deployment
- Load balancing with NGINX for high availability
- Monitoring with Prometheus and Grafana

## Project Structure

```
sentiment-analysis-project/
├── README.md                      # Main documentation file
├── docs/                          # Documentation folder
│   ├── data_preprocessing.md      # Documentation for data preprocessing
│   ├── model_training.md          # Documentation for model training
│   ├── model_evaluation.md        # Documentation for model evaluation
│   ├── deployment.md              # Documentation for deployment
│   └── api_usage.md               # Documentation for API usage
├── src/                           # Source code folder
│   ├── data/                      # Data processing scripts
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # Data preprocessing script
│   │   └── dataset.py             # Dataset creation script
│   ├── models/                    # Model-related scripts
│   │   ├── __init__.py
│   │   ├── model.py               # Model definition script
│   │   └── training.py            # Training script
│   ├── evaluation/                # Evaluation scripts
│   │   ├── __init__.py
│   │   └── metrics.py             # Evaluation metrics script
│   └── api/                       # API scripts
│       ├── __init__.py
│       └── app.py                 # Flask API application
├── scripts/                       # Helper scripts
│   ├── train.py                   # Script to run training
│   ├── evaluate.py                # Script to run evaluation
│   └── run_api.py                 # Script to run API
├── tests/                         # Test scripts
│   ├── __init__.py
│   ├── test_preprocessing.py      # Tests for preprocessing
│   ├── test_model.py              # Tests for model
│   └── test_api.py                # Tests for API
├── config/                        # Configuration files
│   ├── preprocessing_config.json  # Configuration for preprocessing
│   ├── model_config.json          # Configuration for model
│   └── api_config.json            # Configuration for API
├── Dockerfile                     # Dockerfile for containerization
├── docker-compose.yml             # Docker Compose file for deployment
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Docker and Docker Compose (for containerized deployment)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-project.git
   cd sentiment-analysis-project
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK resources:

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

### Data Preprocessing

1. Download the Twitter Entity Sentiment Analysis Dataset from Kaggle.

2. Run the preprocessing script:

   ```bash
   python -m src.data.preprocessing --input data/twitter_training.csv --output data/processed_data
   ```

   This will clean and tokenize the data, and save the processed data to the specified output directory.

### Model Training

1. Configure training parameters in `config/model_config.json`.

2. Run the training script:

   ```bash
   python -m scripts.train --data_dir data/processed_data --output_dir data/models --config ./config/model_config.json
   ```

   This will train the model with the specified hyperparameters and save the best model to the output directory.

### Model Evaluation

1. Run the evaluation script:

   ```bash
   python -m scripts.evaluate --data_dir ./processed_data --model_path ./models/best_model.pt --output_dir ./evaluation
   ```

   This will evaluate the model on the test set and generate comprehensive metrics and visualizations.

### API Deployment

1. Using Docker Compose:

   ```bash
   docker-compose up -d
   ```

   This will build and start the Docker containers for the API, NGINX, Prometheus, and Grafana.

2. Manually:

   ```bash
   python -m src.api.app --model_path ./models/best_model.pt --host 0.0.0.0 --port 5000
   ```

   This will start the Flask API server on the specified host and port.

## API Documentation

### Endpoints

- `GET /api/health`: Health check endpoint
- `POST /api/sentiment`: Sentiment prediction for a single text
- `POST /api/batch-sentiment`: Batch sentiment prediction for multiple texts
- `GET /api/stats`: API statistics

### Example Requests

#### Single Prediction

```bash
curl -X POST http://localhost:5000/api/sentiment \
    -H "Content-Type: application/json" \
    -d '{"text": "I love this new product! It works great."}'
```

Response:

```json
{
  "status": "success",
  "data": {
    "text": "I love this new product! It works great.",
    "cleaned_text": "love new product works great",
    "sentiment": "positive",
    "confidence": 0.9564,
    "probabilities": {
      "negative": 0.0152,
      "neutral": 0.0284,
      "positive": 0.9564
    },
    "processing_time": 0.0512
  }
}
```

#### Batch Prediction

```bash
curl -X POST http://localhost:5000/api/batch-sentiment \
    -H "Content-Type: application/json" \
    -d '{"texts": ["I love this product!", "This is terrible.", "It works as expected."]}'
```

Response:

```json
{
  "status": "success",
  "data": [
    {
      "text": "I love this product!",
      "cleaned_text": "love product",
      "sentiment": "positive",
      "confidence": 0.9732,
      "probabilities": {
        "negative": 0.0098,
        "neutral": 0.0170,
        "positive": 0.9732
      }
    },
    {
      "text": "This is terrible.",
      "cleaned_text": "terrible",
      "sentiment": "negative",
      "confidence": 0.9453,
      "probabilities": {
        "negative": 0.9453,
        "neutral": 0.0428,
        "positive": 0.0119
      }
    },
    {
      "text": "It works as expected.",
      "cleaned_text": "works expected",
      "sentiment": "neutral",
      "confidence": 0.6842,
      "probabilities": {
        "negative": 0.0531,
        "neutral": 0.6842,
        "positive": 0.2627
      }
    }
  ],
  "processing_time": 0.1845
}
```

## Performance

The model achieves the following performance on the test set:

- Accuracy: 0.87
- F1 Score (Macro): 0.86
- Precision (Macro): 0.85
- Recall (Macro): 0.87

The API can handle at least 500 concurrent users with optimized settings. Average response time is around 50ms for single predictions and 120ms for batches of 10 texts.
