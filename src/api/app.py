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
import random
import statistics
import numpy as np
from collections import defaultdict

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

@app.route('/api/load-test', methods=['POST'])
def run_load_test():
    """
    Run a load test on the sentiment analysis API.
    
    Returns:
        JSON: Load test results.
    """
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({'status': 'error', 'message': 'Missing texts field'}), 400
    
    # Get load test parameters
    texts = data['texts']
    users = data.get('users', 10)
    duration = data.get('duration', 30)
    ramp_up = data.get('rampUp', 5)
    batch_size = data.get('batchSize', 1)
    think_time = data.get('thinkTime', 1.0)
    
    # Validate parameters
    if not texts:
        return jsonify({'status': 'error', 'message': 'No texts provided'}), 400
    if users <= 0:
        return jsonify({'status': 'error', 'message': 'Invalid number of users'}), 400
    if duration <= 0:
        return jsonify({'status': 'error', 'message': 'Invalid duration'}), 400
    
    logger.info(f"Starting load test with {users} users for {duration} seconds")
    
    # Initialize results storage
    results = []
    errors = []
    start_time = time.time()
    
    def worker_task():
        """Task for each worker thread."""
        worker_start = time.time()
        worker_end = worker_start + duration
        
        while time.time() < worker_end:
            try:
                batch_texts = random.sample(texts, min(batch_size, len(texts)))
                request_start = time.time()
                
                if batch_size > 1:
                    response = predict_batch(batch_texts)
                    request_type = 'batch'
                else:
                    response = predict_sentiment(batch_texts[0])
                    request_type = 'single'
                
                request_end = time.time()
                
                results.append({
                    'timestamp': request_start - start_time,
                    'response_time': request_end - request_start,
                    'success': True,
                    'request_type': request_type,
                    'batch_size': len(batch_texts)
                })
                
                # Add think time
                time.sleep(max(0.1, random.normalvariate(think_time, think_time / 4)))
            
            except Exception as e:
                errors.append({
                    'timestamp': time.time() - start_time,
                    'error': str(e)
                })
                time.sleep(1)  # Sleep on error
    
    # Start workers with ramp-up
    with ThreadPoolExecutor(max_workers=users) as executor:
        if ramp_up > 0 and users > 1:
            users_per_step = max(1, users // 10)
            step_duration = ramp_up / (users / users_per_step)
            
            futures = []
            active_workers = 0
            
            for i in range(0, users, users_per_step):
                end_index = min(i + users_per_step, users)
                for _ in range(i, end_index):
                    futures.append(executor.submit(worker_task))
                    active_workers += 1
                
                logger.info(f"Started {active_workers}/{users} workers")
                time.sleep(step_duration)
        else:
            # Start all workers at once
            futures = [executor.submit(worker_task) for _ in range(users)]
        
        # Wait for all workers to complete
        for future in futures:
            future.result()
    
    # Calculate metrics
    if not results:
        return jsonify({
            'status': 'error',
            'message': 'No results collected during the test'
        }), 500
    
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r['success'])
    total_duration = time.time() - start_time
    
    # Calculate response time series (1-second buckets)
    time_buckets = defaultdict(list)
    for r in results:
        bucket = int(r['timestamp'])
        time_buckets[bucket].append(r['response_time'])
    
    response_time_series = []
    for t in range(int(total_duration) + 1):
        times = time_buckets.get(t, [])
        if times:
            response_time_series.append({
                'timestamp': t,
                'mean': statistics.mean(times),
                'p95': np.percentile(times, 95) if len(times) > 1 else times[0]
            })
    
    # Prepare results
    test_results = {
        'total_requests': total_requests,
        'successful_requests': successful_requests,
        'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
        'total_errors': len(errors),
        'test_duration': total_duration,
        'throughput': total_requests / total_duration,
        'response_time': {
            'min': min(r['response_time'] for r in results),
            'max': max(r['response_time'] for r in results),
            'mean': statistics.mean(r['response_time'] for r in results),
            'median': statistics.median(r['response_time'] for r in results),
            'p95': np.percentile([r['response_time'] for r in results], 95),
            'p99': np.percentile([r['response_time'] for r in results], 99)
        },
        'response_time_series': response_time_series
    }
    
    logger.info(f"Load test completed: {total_requests} requests, "
                f"{test_results['success_rate']:.1f}% success rate, "
                f"{test_results['throughput']:.2f} req/s")
    
    return jsonify({
        'status': 'success',
        'data': test_results
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