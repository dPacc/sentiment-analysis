#!/usr/bin/env python
"""
Script to test the sentiment analysis API.

This script sends requests to the API and displays the responses,
allowing users to test the API functionality.
"""

import argparse
import json
import time
import logging
import requests
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test sentiment analysis API')
    parser.add_argument('--url', default='http://localhost', help='Base URL of the API')
    parser.add_argument('--port', type=int, default=80, help='Port of the API')
    parser.add_argument('--endpoint', default='/api/sentiment', help='API endpoint to test')
    parser.add_argument('--text', help='Text to analyze')
    parser.add_argument('--file', help='File containing texts to analyze, one per line')
    parser.add_argument('--batch', action='store_true', help='Use batch endpoint for multiple texts')
    parser.add_argument('--concurrent', type=int, default=1, help='Number of concurrent requests')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
    parser.add_argument('--delay', type=float, default=0, help='Delay between requests in seconds')
    
    return parser.parse_args()

def test_health(base_url):
    """Test the health check endpoint."""
    url = f"{base_url}/api/health"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        logger.info(f"Health check: {data['status']}")
        logger.info(f"Message: {data['message']}")
        
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking health: {e}")
        return False

def test_single(base_url, endpoint, text):
    """Test the sentiment analysis endpoint with a single text."""
    url = f"{base_url}{endpoint}"
    
    try:
        # Prepare request
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers)
        elapsed_time = time.time() - start_time
        
        # Parse response
        data = response.json()
        
        # Display results
        if response.status_code == 200 and data['status'] == 'success':
            result = data['data']
            logger.info(f"Text: {result['text'][:50]}...")
            logger.info(f"Cleaned text: {result['cleaned_text'][:50]}...")
            logger.info(f"Sentiment: {result['sentiment']}")
            logger.info(f"Confidence: {result['confidence']:.4f}")
            logger.info(f"Processing time: {result['processing_time']:.4f}s")
            logger.info(f"Total request time: {elapsed_time:.4f}s")
            
            return True
        else:
            logger.error(f"Error: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error testing API: {e}")
        return False

def test_batch(base_url, texts):
    """Test the batch sentiment analysis endpoint."""
    url = f"{base_url}/api/batch-sentiment"
    
    try:
        # Prepare request
        payload = {"texts": texts}
        headers = {"Content-Type": "application/json"}
        
        # Send request
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers)
        elapsed_time = time.time() - start_time
        
        # Parse response
        data = response.json()
        
        # Display results
        if response.status_code == 200 and data['status'] == 'success':
            results = data['data']
            logger.info(f"Processed {len(results)} texts in {data['processing_time']:.4f}s")
            logger.info(f"Total request time: {elapsed_time:.4f}s")
            
            # Display summary
            sentiments = {}
            for result in results:
                sentiment = result['sentiment']
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
            
            logger.info("Sentiment distribution:")
            for sentiment, count in sentiments.items():
                logger.info(f"  {sentiment}: {count} ({count/len(results)*100:.1f}%)")
            
            return True
        else:
            logger.error(f"Error: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error testing batch API: {e}")
        return False

def load_texts_from_file(file_path):
    """Load texts from a file, one per line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading texts from file: {e}")
        return []

def run_concurrent_tests(base_url, endpoint, texts, batch, concurrent, iterations, delay):
    """Run concurrent tests against the API."""
    total_requests = len(texts) * iterations
    successful_requests = 0
    total_time = 0
    
    logger.info(f"Running {concurrent} concurrent workers for {total_requests} requests")
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = []
        
        # Create tasks
        for i in range(iterations):
            if batch:
                # Submit batch request
                future = executor.submit(test_batch, base_url, texts)
                futures.append(future)
                time.sleep(delay)  # Delay between batches
            else:
                # Submit individual requests
                for text in texts:
                    future = executor.submit(test_single, base_url, endpoint, text)
                    futures.append(future)
                    time.sleep(delay)  # Delay between requests
        
        # Process results
        start_time = time.time()
        for future in as_completed(futures):
            result = future.result()
            if result:
                successful_requests += 1
        
        total_time = time.time() - start_time
    
    # Display summary
    success_rate = (successful_requests / total_requests) * 100
    requests_per_second = total_requests / total_time if total_time > 0 else 0
    
    logger.info(f"Test completed: {successful_requests}/{total_requests} successful ({success_rate:.1f}%)")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Requests per second: {requests_per_second:.2f}")
    
    return success_rate >= 99  # Consider test successful if ≥99% requests succeeded

def main():
    """Main function to test the API."""
    args = parse_args()
    
    # Build base URL
    base_url = f"{args.url}:{args.port}" if args.port != 80 else args.url
    
    # Check API health
    if not test_health(base_url):
        logger.error("API health check failed. Make sure the API is running.")
        sys.exit(1)
    
    # Get texts to analyze
    texts = []
    if args.file:
        texts = load_texts_from_file(args.file)
        if not texts:
            logger.error("No texts loaded from file.")
            sys.exit(1)
    elif args.text:
        texts = [args.text]
    else:
        logger.error("Either --text or --file must be specified.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(texts)} texts for testing")
    
    # Run tests
    if args.concurrent > 1 or args.iterations > 1:
        success = run_concurrent_tests(
            base_url=base_url,
            endpoint=args.endpoint,
            texts=texts,
            batch=args.batch,
            concurrent=args.concurrent,
            iterations=args.iterations,
            delay=args.delay
        )
    else:
        if args.batch and len(texts) > 1:
            success = test_batch(base_url, texts)
        else:
            success = all(test_single(base_url, args.endpoint, text) for text in texts)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()