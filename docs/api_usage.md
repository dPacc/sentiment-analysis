# API Usage Guide

This document provides comprehensive guidance on using the Twitter sentiment analysis API, including endpoint descriptions, request/response formats, and usage examples.

## API Overview

The sentiment analysis API provides a simple and efficient way to analyze the sentiment of text content. The API classifies text into three sentiment categories:

- **Positive**: Text expresses a favorable or positive opinion
- **Neutral**: Text expresses a neutral or factual statement
- **Negative**: Text expresses an unfavorable or negative opinion

The API supports both single text analysis and batch processing for multiple texts.

## Base URL

When deployed locally, the API is accessible at:

```
http://localhost:80/api
```

For production deployments, replace `localhost` with your server's domain name or IP address.

## Authentication

Currently, the API uses IP-based access control and does not require authentication tokens. For production deployments, it is recommended to implement API key authentication.

## Rate Limits

The API implements the following rate limits:

- 10 requests per second per IP address
- Burst capacity of 20 requests

If you exceed these limits, you will receive a 429 (Too Many Requests) response.

## Endpoints

### 1. Health Check

Check if the API service is running properly.

**Endpoint:** `GET /api/health`

**Response:**

```json
{
  "status": "healthy",
  "message": "API is running"
}
```

**Status Codes:**
- 200: API is healthy
- 503: API is unhealthy (model not loaded)

### 2. Single Text Sentiment Analysis

Analyze the sentiment of a single text.

**Endpoint:** `POST /api/sentiment`

**Request Body:**

```json
{
  "text": "I absolutely love this new feature! It makes everything so much easier."
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "text": "I absolutely love this new feature! It makes everything so much easier.",
    "cleaned_text": "absolutely love new feature makes everything much easier",
    "sentiment": "positive",
    "confidence": 0.9651,
    "probabilities": {
      "negative": 0.0123,
      "neutral": 0.0226,
      "positive": 0.9651
    },
    "processing_time": 0.0435
  }
}
```

**Status Codes:**
- 200: Successful analysis
- 400: Bad request (missing or invalid text)
- 500: Server error

### 3. Batch Sentiment Analysis

Analyze the sentiment of multiple texts in a single request.

**Endpoint:** `POST /api/batch-sentiment`

**Request Body:**

```json
{
  "texts": [
    "I absolutely love this new feature! It makes everything so much easier.",
    "The service was terrible and the staff was rude.",
    "The product arrived on time as expected."
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "data": [
    {
      "text": "I absolutely love this new feature! It makes everything so much easier.",
      "cleaned_text": "absolutely love new feature makes everything much easier",
      "sentiment": "positive",
      "confidence": 0.9651,
      "probabilities": {
        "negative": 0.0123,
        "neutral": 0.0226,
        "positive": 0.9651
      }
    },
    {
      "text": "The service was terrible and the staff was rude.",
      "cleaned_text": "service terrible staff rude",
      "sentiment": "negative",
      "confidence": 0.9324,
      "probabilities": {
        "negative": 0.9324,
        "neutral": 0.0542,
        "positive": 0.0134
      }
    },
    {
      "text": "The product arrived on time as expected.",
      "cleaned_text": "product arrived time expected",
      "sentiment": "neutral",
      "confidence": 0.7823,
      "probabilities": {
        "negative": 0.0513,
        "neutral": 0.7823,
        "positive": 0.1664
      }
    }
  ],
  "processing_time": 0.1567
}
```

**Status Codes:**
- 200: Successful analysis
- 400: Bad request (missing or invalid texts array)
- 500: Server error

### 4. API Statistics

Get usage statistics for the API.

**Endpoint:** `GET /api/stats`

**Response:**

```json
{
  "status": "success",
  "data": {
    "requests_processed": 1453,
    "model_device": "cuda:0",
    "server_time": "2023-06-15 14:32:45"
  }
}
```

**Status Codes:**
- 200: Successful request

## Response Fields Explained

The API returns the following fields in the response:

- **text**: The original input text
- **cleaned_text**: The preprocessed text after cleaning
- **sentiment**: The predicted sentiment label (positive, neutral, or negative)
- **confidence**: The probability score for the predicted sentiment
- **probabilities**: The probability scores for all sentiment classes
- **processing_time**: The time taken to process the request (in seconds)

## Usage Examples

### cURL Examples

#### Health Check

```bash
curl -X GET http://localhost/api/health
```

#### Single Text Analysis

```bash
curl -X POST http://localhost/api/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this new feature! It makes everything so much easier."}'
```

#### Batch Text Analysis

```bash
curl -X POST http://localhost/api/batch-sentiment \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I absolutely love this new feature!", "The service was terrible.", "The product arrived on time."]}'
```

#### API Statistics

```bash
curl -X GET http://localhost/api/stats
```

### Python Examples

#### Single Text Analysis

```python
import requests
import json

url = "http://localhost/api/sentiment"
headers = {"Content-Type": "application/json"}
data = {"text": "I absolutely love this new feature! It makes everything so much easier."}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

print(f"Sentiment: {result['data']['sentiment']}")
print(f"Confidence: {result['data']['confidence']:.2f}")
```

#### Batch Text Analysis

```python
import requests
import json

url = "http://localhost/api/batch-sentiment"
headers = {"Content-Type": "application/json"}
data = {
    "texts": [
        "I absolutely love this new feature!",
        "The service was terrible.",
        "The product arrived on time."
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

for i, item in enumerate(result['data']):
    print(f"Text {i+1}: {item['text']}")
    print(f"Sentiment: {item['sentiment']}")
    print(f"Confidence: {item['confidence']:.2f}")
    print()
```

### JavaScript Examples

#### Single Text Analysis

```javascript
async function analyzeSentiment(text) {
  const response = await fetch('http://localhost/api/sentiment', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });
  
  const result = await response.json();
  
  if (result.status === 'success') {
    console.log(`Sentiment: ${result.data.sentiment}`);
    console.log(`Confidence: ${result.data.confidence.toFixed(2)}`);
  } else {
    console.error(`Error: ${result.message}`);
  }
}

analyzeSentiment('I absolutely love this new feature! It makes everything so much easier.');
```

#### Batch Text Analysis

```javascript
async function analyzeBatchSentiment(texts) {
  const response = await fetch('http://localhost/api/batch-sentiment', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ texts }),
  });
  
  const result = await response.json();
  
  if (result.status === 'success') {
    result.data.forEach((item, index) => {
      console.log(`Text ${index + 1}: ${item.text}`);
      console.log(`Sentiment: ${item.sentiment}`);
      console.log(`Confidence: ${item.confidence.toFixed(2)}`);
      console.log();
    });
  } else {
    console.error(`Error: ${result.message}`);
  }
}

analyzeBatchSentiment([
  'I absolutely love this new feature!',
  'The service was terrible.',
  'The product arrived on time.'
]);
```

## Best Practices

### Optimizing Performance

1. **Use batch processing** for multiple texts to reduce overhead and improve throughput.
2. **Limit text length** to 512 characters for optimal performance.
3. **Implement caching** for frequently analyzed texts to reduce API calls.

### Handling Errors

1. **Check response status** to ensure successful processing.
2. **Implement retry logic** with exponential backoff for temporary failures.
3. **Handle rate limiting** by respecting the 429 status code and backing off.

### Interpreting Results

1. **Consider confidence scores** when making decisions based on sentiment analysis.
2. **Low confidence scores** (below 0.6) may indicate ambiguous sentiment.
3. **Compare sentiment probabilities** to understand how confident the model is in its prediction.

## Testing the API

A simple way to test the API is to use the provided Python script:

```bash
python -m scripts.test_api --url http://localhost --endpoint /api/sentiment --text "I love this product!"
```

This script sends a request to the API and displays the response.

## Error Handling

The API returns errors in the following format:

```json
{
  "status": "error",
  "message": "Error message description"
}
```

Common error messages include:

- "Missing text field": The required text field is missing in the request.
- "Missing texts field": The required texts array is missing in the batch request.
- "Texts field must be a list": The texts field in the batch request is not an array.
- "Error processing request": An internal server error occurred during processing.

## Support and Feedback

For support or to provide feedback on the API, please open an issue on the GitHub repository or contact the development team.