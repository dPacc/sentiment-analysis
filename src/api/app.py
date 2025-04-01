"""
Flask API for Twitter sentiment analysis.

This module implements a Flask API for serving the sentiment analysis model.
"""

from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
import logging
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Local imports
from ..models.model import load_model
from ..data.preprocessing import clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Global variables
model = None
tokenizer = None
device = None
label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
request_counter = 0
request_lock = threading.Lock()
max_workers = 10  # Number of worker threads for parallel processing

def load_resources(model_path, device_name='cuda'):
    """
    Load model and tokenizer.
    
    Args:
        model_path (str): Path to the model file.
        device_name (str): Device to load the model on ('cuda' or 'cpu').
    """
    global model, tokenizer, device
    
    logger.info("Loading resources")
    
    # Set device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(model_path, device)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    logger.info("Resources loaded successfully")

def predict_sentiment(text):
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Input text.
        
    Returns:
        dict: Dictionary containing prediction results.
    """
    global model, tokenizer, device, label_mapping
    
    # Clean text
    cleaned_text = clean_text(text, remove_stopwords=True)
    
    # Tokenize text
    encoded_input = tokenizer(
        cleaned_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Move input to device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Get prediction
    with torch.no_grad():
        model.eval()
        outputs = model(**encoded_input)
        
        # Get prediction
        if isinstance(outputs, tuple):
            logits = outputs[1]
        else:
            logits = outputs
        
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    
    # Convert to labels
    sentiment = label_mapping[prediction]
    probabilities = probs[0].cpu().numpy().tolist()
    
    # Create result
    result = {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'confidence': probabilities[prediction],
        'probabilities': {label_mapping[i]: prob for i, prob in enumerate(probabilities)}
    }
    
    return result

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON: Health status.
    """
    if model is None or tokenizer is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    
    return jsonify({'status': 'healthy', 'message': 'API is running'}), 200

@app.route('/api/sentiment', methods=['POST'])
def predict():
    """
    Sentiment prediction endpoint.
    
    Returns:
        JSON: Prediction results.
    """
    global request_counter
    
    # Track requests
    with request_lock:
        request_counter += 1
        current_request = request_counter
    
    # Get data from request
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'status': 'error', 'message': 'Missing text field'}), 400
    
    # Log request
    logger.info(f"Received request #{current_request}: {data['text'][:50]}...")
    
    try:
        # Get prediction
        start_time = time.time()
        result = predict_sentiment(data['text'])
        elapsed_time = time.time() - start_time
        
        # Add timing information
        result['processing_time'] = elapsed_time
        
        # Log response
        logger.info(f"Request #{current_request} processed in {elapsed_time:.4f}s: {result['sentiment']}")
        
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
    except Exception as e:
        logger.error(f"Error processing request #{current_request}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/batch-sentiment', methods=['POST'])
def batch_predict():
    """
    Batch sentiment prediction endpoint.
    
    Returns:
        JSON: Batch prediction results.
    """
    # Get data from request
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({'status': 'error', 'message': 'Missing texts field'}), 400
    
    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({'status': 'error', 'message': 'Texts field must be a list'}), 400
    
    # Log request
    logger.info(f"Received batch request with {len(texts)} texts")
    
    try:
        # Process texts in parallel
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(predict_sentiment, texts))
        
        elapsed_time = time.time() - start_time
        
        # Log response
        logger.info(f"Batch request processed in {elapsed_time:.4f}s ({len(texts) / elapsed_time:.2f} texts/s)")
        
        return jsonify({
            'status': 'success',
            'data': results,
            'processing_time': elapsed_time
        }), 200
    except Exception as e:
        logger.error(f"Error processing batch request: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """
    API statistics endpoint.
    
    Returns:
        JSON: API statistics.
    """
    global request_counter
    
    stats = {
        'requests_processed': request_counter,
        'model_device': str(device),
        'server_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return jsonify({
        'status': 'success',
        'data': stats
    }), 200

def start_server(model_path, host='0.0.0.0', port=5000, debug=False, device_name='cuda'):
    """
    Start the Flask API server.
    
    Args:
        model_path (str): Path to the model file.
        host (str): Host to bind the server.
        port (int): Port to bind the server.
        debug (bool): Whether to run in debug mode.
        device_name (str): Device to run the model on ('cuda' or 'cpu').
    """
    # Load resources
    load_resources(model_path, device_name)
    
    # Start server
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Start sentiment analysis API')
    parser.add_argument('--model_path', required=True, help='Path to the model file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to run the model on')
    
    args = parser.parse_args()
    
    # Start server
    start_server(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        debug=args.debug,
        device_name=args.device
    )