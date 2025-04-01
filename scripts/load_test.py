#!/usr/bin/env python
"""
Script to perform load testing on the sentiment analysis API.

This script simulates high traffic to the API and measures its performance
under load, including response time, throughput, and error rate.
"""

import argparse
import json
import time
import logging
import random
import statistics
import requests
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load test sentiment analysis API')
    parser.add_argument('--url', default='http://localhost', help='Base URL of the API')
    parser.add_argument('--port', type=int, default=80, help='Port of the API')
    parser.add_argument('--file', required=True, help='File containing texts to analyze, one per line')
    parser.add_argument('--users', type=int, default=50, help='Number of simulated concurrent users')
    parser.add_argument('--duration', type=int, default=60, help='Duration of the test in seconds')
    parser.add_argument('--ramp-up', type=int, default=10, help='Ramp-up period in seconds')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of texts in a batch (1 for single endpoint)')
    parser.add_argument('--think-time', type=float, default=1.0, help='Think time between requests in seconds')
    parser.add_argument('--output', default='load_test_results', help='Output directory for results')
    
    return parser.parse_args()

def load_texts_from_file(file_path):
    """Load texts from a file, one per line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading texts from file: {e}")
        return []

class LoadTestWorker:
    """Worker class to simulate a user making requests to the API."""
    
    def __init__(self, worker_id, base_url, texts, batch_size, think_time):
        """
        Initialize a worker.
        
        Args:
            worker_id (int): Unique identifier for the worker.
            base_url (str): Base URL of the API.
            texts (list): List of texts to analyze.
            batch_size (int): Number of texts in a batch.
            think_time (float): Think time between requests in seconds.
        """
        self.worker_id = worker_id
        self.base_url = base_url
        self.texts = texts
        self.batch_size = batch_size
        self.think_time = think_time
        self.results = []
        self.errors = []
        self.is_running = True
    
    def run(self, duration):
        """
        Run the worker for the specified duration.
        
        Args:
            duration (int): Duration in seconds.
        """
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time and self.is_running:
            try:
                # Select random texts for the request
                if self.batch_size > 1:
                    # Batch request
                    selected_texts = random.choices(self.texts, k=self.batch_size)
                    result = self.make_batch_request(selected_texts)
                else:
                    # Single request
                    selected_text = random.choice(self.texts)
                    result = self.make_single_request(selected_text)
                
                # Add result
                self.results.append(result)
                
                # Think time (with randomization)
                think_time = max(0.1, random.normalvariate(self.think_time, self.think_time / 4))
                time.sleep(think_time)
            
            except Exception as e:
                error = {
                    'timestamp': time.time(),
                    'error': str(e)
                }
                self.errors.append(error)
                time.sleep(1)  # Sleep on error to avoid flooding
        
        return self.results, self.errors
    
    def make_single_request(self, text):
        """
        Make a single request to the sentiment analysis API.
        
        Args:
            text (str): Text to analyze.
            
        Returns:
            dict: Result of the request.
        """
        url = f"{self.base_url}/api/sentiment"
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        # Record start time
        start_time = time.time()
        
        # Send request
        response = requests.post(url, json=payload, headers=headers)
        
        # Record end time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Parse response
        data = response.json()
        
        # Create result
        result = {
            'worker_id': self.worker_id,
            'timestamp': start_time,
            'response_time': response_time,
            'status_code': response.status_code,
            'success': response.status_code == 200 and data['status'] == 'success',
            'request_type': 'single',
            'text_length': len(text)
        }
        
        if result['success']:
            result['sentiment'] = data['data']['sentiment']
            result['confidence'] = data['data']['confidence']
            result['processing_time'] = data['data'].get('processing_time', 0)
        
        return result
    
    def make_batch_request(self, texts):
        """
        Make a batch request to the sentiment analysis API.
        
        Args:
            texts (list): List of texts to analyze.
            
        Returns:
            dict: Result of the request.
        """
        url = f"{self.base_url}/api/batch-sentiment"
        payload = {"texts": texts}
        headers = {"Content-Type": "application/json"}
        
        # Record start time
        start_time = time.time()
        
        # Send request
        response = requests.post(url, json=payload, headers=headers)
        
        # Record end time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Parse response
        data = response.json()
        
        # Create result
        result = {
            'worker_id': self.worker_id,
            'timestamp': start_time,
            'response_time': response_time,
            'status_code': response.status_code,
            'success': response.status_code == 200 and data['status'] == 'success',
            'request_type': 'batch',
            'batch_size': len(texts),
            'total_text_length': sum(len(text) for text in texts)
        }
        
        if result['success']:
            result['processing_time'] = data.get('processing_time', 0)
            result['sentiments'] = [item['sentiment'] for item in data['data']]
        
        return result

class LoadTest:
    """Class to manage the load test."""
    
    def __init__(self, args):
        """
        Initialize the load test.
        
        Args:
            args: Command line arguments.
        """
        self.base_url = f"{args.url}:{args.port}" if args.port != 80 else args.url
        self.texts = load_texts_from_file(args.file)
        self.users = args.users
        self.duration = args.duration
        self.ramp_up = args.ramp_up
        self.batch_size = args.batch_size
        self.think_time = args.think_time
        self.output_dir = args.output
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save test configuration
        self.config = vars(args)
        self.config['test_id'] = datetime.now().strftime('%Y%m%d%H%M%S')
        self.config['num_texts'] = len(self.texts)
        
        config_path = os.path.join(self.output_dir, 'test_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def run(self):
        """Run the load test."""
        logger.info(f"Starting load test with {self.users} users for {self.duration} seconds")
        logger.info(f"Using {len(self.texts)} unique texts, batch size {self.batch_size}")
        
        # Check API health
        if not self._check_health():
            logger.error("API health check failed. Make sure the API is running.")
            return False
        
        # Create workers
        workers = []
        for i in range(self.users):
            worker = LoadTestWorker(
                worker_id=i,
                base_url=self.base_url,
                texts=self.texts,
                batch_size=self.batch_size,
                think_time=self.think_time
            )
            workers.append(worker)
        
        # Start test with ramp-up
        results = []
        errors = []
        
        if self.ramp_up > 0 and self.users > 1:
            users_per_step = max(1, self.users // 10)
            step_duration = self.ramp_up / (self.users / users_per_step)
            
            logger.info(f"Ramping up {users_per_step} users every {step_duration:.1f} seconds")
            
            active_workers = 0
            with ThreadPoolExecutor(max_workers=self.users) as executor:
                futures = []
                
                # Start workers gradually
                start_time = time.time()
                for i in range(0, self.users, users_per_step):
                    end_index = min(i + users_per_step, self.users)
                    for j in range(i, end_index):
                        future = executor.submit(workers[j].run, self.duration)
                        futures.append(future)
                        active_workers += 1
                    
                    logger.info(f"Started {active_workers}/{self.users} users")
                    time.sleep(step_duration)
                
                # Collect results
                for future in as_completed(futures):
                    worker_results, worker_errors = future.result()
                    results.extend(worker_results)
                    errors.extend(worker_errors)
        else:
            # Start all workers at once
            with ThreadPoolExecutor(max_workers=self.users) as executor:
                futures = [executor.submit(worker.run, self.duration) for worker in workers]
                
                # Collect results
                for future in as_completed(futures):
                    worker_results, worker_errors = future.result()
                    results.extend(worker_results)
                    errors.extend(worker_errors)
        
        # Analyze and save results
        self._analyze_results(results, errors)
        
        return True
    
    def _check_health(self):
        """Check the health of the API."""
        url = f"{self.base_url}/api/health"
        
        try:
            response = requests.get(url)
            return response.status_code == 200
        except Exception:
            return False
    
    def _analyze_results(self, results, errors):
        """
        Analyze test results and save them.
        
        Args:
            results (list): List of request results.
            errors (list): List of errors.
        """
        if not results:
            logger.error("No results collected during the test.")
            return
        
        # Save raw results
        results_path = os.path.join(self.output_dir, 'raw_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        if errors:
            errors_path = os.path.join(self.output_dir, 'errors.json')
            with open(errors_path, 'w') as f:
                json.dump(errors, f)
        
        # Calculate metrics
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        
        response_times = [r['response_time'] for r in results]
        
        # Calculate summary statistics
        summary = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': success_rate,
            'total_errors': len(errors),
            'test_duration': self.duration,
            'throughput': total_requests / self.duration,
            'response_time': {
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'mean': statistics.mean(response_times) if response_times else 0,
                'median': statistics.median(response_times) if response_times else 0,
                'percentile_90': np.percentile(response_times, 90) if response_times else 0,
                'percentile_95': np.percentile(response_times, 95) if response_times else 0,
                'percentile_99': np.percentile(response_times, 99) if response_times else 0,
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        }
        
        # Add batch-specific metrics if using batch requests
        if self.batch_size > 1:
            batch_requests = [r for r in results if r['request_type'] == 'batch']
            if batch_requests:
                processing_times = [r.get('processing_time', 0) for r in batch_requests if 'processing_time' in r]
                summary['batch_requests'] = {
                    'count': len(batch_requests),
                    'total_texts_processed': sum(r.get('batch_size', 0) for r in batch_requests),
                    'processing_time': {
                        'min': min(processing_times) if processing_times else 0,
                        'max': max(processing_times) if processing_times else 0,
                        'mean': statistics.mean(processing_times) if processing_times else 0,
                        'median': statistics.median(processing_times) if processing_times else 0
                    }
                }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Log summary
        logger.info(f"Test completed with {total_requests} requests ({success_rate:.1f}% success rate)")
        logger.info(f"Throughput: {summary['throughput']:.2f} requests/second")
        logger.info(f"Response time (mean): {summary['response_time']['mean']:.4f}s")
        logger.info(f"Response time (95th percentile): {summary['response_time']['percentile_95']:.4f}s")
        
        # Generate plots
        self._generate_plots(results)
    
    def _generate_plots(self, results):
        """
        Generate visualizations of test results.
        
        Args:
            results (list): List of request results.
        """
        # Sort results by timestamp
        results.sort(key=lambda r: r['timestamp'])
        
        # Calculate timestamps relative to test start
        start_time = results[0]['timestamp']
        for r in results:
            r['relative_time'] = r['timestamp'] - start_time
        
        # Create time buckets (1-second intervals)
        buckets = defaultdict(list)
        for r in results:
            bucket = int(r['relative_time'])
            buckets[bucket].append(r)
        
        # Calculate throughput per second
        throughput_data = [(t, len(reqs)) for t, reqs in sorted(buckets.items())]
        
        # Calculate response time statistics per second
        response_time_data = []
        for t, reqs in sorted(buckets.items()):
            response_times = [r['response_time'] for r in reqs]
            if response_times:
                mean_rt = statistics.mean(response_times)
                median_rt = statistics.median(response_times)
                p95_rt = np.percentile(response_times, 95)
                response_time_data.append((t, mean_rt, median_rt, p95_rt))
        
        # Calculate success rate per second
        success_data = []
        for t, reqs in sorted(buckets.items()):
            success_count = sum(1 for r in reqs if r['success'])
            total_count = len(reqs)
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            success_data.append((t, success_rate))
        
        # Create figure with subplots
        plt.figure(figsize=(12, 18))
        
        # Plot throughput
        plt.subplot(3, 1, 1)
        times, throughputs = zip(*throughput_data) if throughput_data else ([], [])
        plt.plot(times, throughputs)
        plt.title('Throughput (Requests per Second)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Requests')
        plt.grid(True)
        
        # Plot response times
        plt.subplot(3, 1, 2)
        if response_time_data:
            times, means, medians, p95s = zip(*response_time_data)
            plt.plot(times, means, label='Mean')
            plt.plot(times, medians, label='Median')
            plt.plot(times, p95s, label='95th Percentile')
            plt.title('Response Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Response Time (seconds)')
            plt.legend()
            plt.grid(True)
        
        # Plot success rate
        plt.subplot(3, 1, 3)
        times, rates = zip(*success_data) if success_data else ([], [])
        plt.plot(times, rates)
        plt.title('Success Rate')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 105)
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.output_dir, 'load_test_plots.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Create histogram of response times
        plt.figure(figsize=(10, 6))
        response_times = [r['response_time'] for r in results]
        plt.hist(response_times, bins=50)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Count')
        plt.grid(True)
        
        # Save histogram
        hist_path = os.path.join(self.output_dir, 'response_time_histogram.png')
        plt.savefig(hist_path)
        plt.close()

def main():
    """Main function to run the load test."""
    args = parse_args()
    
    # Validate arguments
    if args.users <= 0:
        logger.error("Number of users must be greater than 0.")
        sys.exit(1)
    
    if args.duration <= 0:
        logger.error("Duration must be greater than 0.")
        sys.exit(1)
    
    if args.batch_size <= 0:
        logger.error("Batch size must be greater than 0.")
        sys.exit(1)
    
    # Load texts
    texts = load_texts_from_file(args.file)
    if not texts:
        logger.error("No texts loaded from file.")
        sys.exit(1)
    
    # Run load test
    load_test = LoadTest(args)
    success = load_test.run()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Script to perform load testing on the sentiment analysis API.

This script simulates high traffic to the API and measures its performance
under load, including response time, throughput, and error rate.
"""

import argparse
import json
import time
import logging
import random
import statistics
import requests
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Load test sentiment analysis API')
    parser.add_argument('--url', default='http://localhost', help='Base URL of the API')
    parser.add_argument('--port', type=int, default=80, help='Port of the API')
    parser.add_argument('--file', required=True, help='File containing texts to analyze, one per line')
    parser.add_argument('--users', type=int, default=50, help='Number of simulated concurrent users')
    parser.add_argument('--duration', type=int, default=60, help='Duration of the test in seconds')
    parser.add_argument('--ramp-up', type=int, default=10, help='Ramp-up period in seconds')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of texts in a batch (1 for single endpoint)')
    parser.add_argument('--think-time', type=float, default=1.0, help='Think time between requests in seconds')
    parser.add_argument('--output', default='load_test_results', help='Output directory for results')
    
    return parser.parse_args()

def load_texts_from_file(file_path):
    """Load texts from a file, one per line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading texts from file: {e}")
        return []

class LoadTestWorker:
    """Worker class to simulate a user making requests to the API."""
    
    def __init__(self, worker_id, base_url, texts, batch_size, think_time):
        """
        Initialize a worker.
        
        Args:
            worker_id (int): Unique identifier for the worker.
            base_url (str): Base URL of the API.
            texts (list): List of texts to analyze.
            batch_size (int): Number of texts in a batch.
            think_time (float): Think time between requests in seconds.
        """
        self.worker_id = worker_id
        self.base_url = base_url
        self.texts = texts
        self.batch_size = batch_size
        self.think_time = think_time
        self.results = []
        self.errors = []
        self.is_running = True
    
    def run(self, duration):
        """
        Run the worker for the specified duration.
        
        Args:
            duration (int): Duration in seconds.
        """
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time and self.is_running:
            try:
                # Select random texts for the request
                if self.batch_size > 1:
                    # Batch request
                    selected_texts = random.choices(self.texts, k=self.batch_size)
                    result = self.make_batch_request(selected_texts)
                else:
                    # Single request
                    selected_text = random.choice(self.texts)
                    result = self.make_single_request(selected_text)
                
                # Add result
                self.results.append(result)
                
                # Think time (with randomization)
                think_time = max(0.1, random.normalvariate(self.think_time, self.think_time / 4))
                time.sleep(think_time)
            
            except Exception as e:
                error = {
                    'timestamp': time.time(),
                    'error': str(e)
                }
                self.errors.append(error)
                time.sleep(1)  # Sleep on error to avoid flooding
        
        return self.results, self.errors
    
    def make_single_request(self, text):
        """
        Make a single request to the sentiment analysis API.
        
        Args:
            text (str): Text to analyze.
            
        Returns:
            dict: Result of the request.
        """
        url = f"{self.base_url}/api/sentiment"
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        # Record start time
        start_time = time.time()
        
        # Send request
        response = requests.post(url, json=payload, headers=headers)
        
        # Record end time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Parse response
        data = response.json()
        
        # Create result
        result = {
            'worker_id': self.worker_id,
            'timestamp': start_time,
            'response_time': response_time,
            'status_code': response.status_code,
            'success': response.status_code == 200 and data['status'] == 'success',
            'request_type': 'single',
            'text_length': len(text)
        }
        
        if result['success']:
            result['sentiment'] = data['data']['sentiment']
            result['confidence'] = data['data']['confidence']
            result['processing_time'] = data['data'].get('processing_time', 0)
        
        return result
    
    def make_batch_request(self, texts):
        """
        Make a batch request to the sentiment analysis API.
        
        Args:
            texts (list): List of texts to analyze.
            
        Returns:
            dict: Result of the request.
        """
        url = f"{self.base_url}/api/batch-sentiment"
        payload = {"texts": texts}
        headers = {"Content-Type": "application/json"}
        
        # Record start time
        start_time = time.time()
        
        # Send request
        response = requests.post(url, json=payload, headers=headers)
        
        # Record end time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Parse response
        data = response.json()
        
        # Create result
        result = {
            'worker_id': self.worker_id,
            'timestamp': start_time,
            'response_time': response_time,
            'status_code': response.status_code,
            'success': response.status_code == 200 and data['status'] == 'success',
            'request_type': 'batch',
            'batch_size': len(texts),
            'total_text_length': sum(len(text) for text in texts)
        }
        
        if result['success']:
            result['processing_time'] = data.get('processing_time', 0)
            result['sentiments'] = [item['sentiment'] for item in data['data']]
        
        return result

class LoadTest:
    """Class to manage the load test."""
    
    def __init__(self, args):
        """
        Initialize the load test.
        
        Args:
            args: Command line arguments.
        """
        self.base_url = f"{args.url}:{args.port}" if args.port != 80 else args.url
        self.texts = load_texts_from_file(args.file)
        self.users = args.users
        self.duration = args.duration
        self.ramp_up = args.ramp_up
        self.batch_size = args.batch_size
        self.think_time = args.think_time
        self.output_dir = args.output
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save test configuration
        self.config = vars(args)
        self.config['test_id'] = datetime.now().strftime('%Y%m%d%H%M%S')
        self.config['num_texts'] = len(self.texts)
        
        config_path = os.path.join(self.output_dir, 'test_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def run(self):
        """Run the load test."""
        logger.info(f"Starting load test with {self.users} users for {self.duration} seconds")
        logger.info(f"Using {len(self.texts)} unique texts, batch size {self.batch_size}")
        
        # Check API health
        if not self._check_health():
            logger.error("API health check failed. Make sure the API is running.")
            return False
        
        # Create workers
        workers = []
        for i in range(self.users):
            worker = LoadTestWorker(
                worker_id=i,
                base_url=self.base_url,
                texts=self.texts,
                batch_size=self.batch_size,
                think_time=self.think_time
            )
            workers.append(worker)
        
        # Start test with ramp-up
        results = []
        errors = []
        
        if self.ramp_up > 0 and self.users > 1:
            users_per_step = max(1, self.users // 10)
            step_duration = self.ramp_up / (self.users / users_per_step)
            
            logger.info(f"Ramping up {users_per_step} users every {step_duration:.1f} seconds")
            
            active_workers = 0
            with ThreadPoolExecutor(max_workers=self.users) as executor:
                futures = []
                
                # Start workers gradually
                start_time = time.time()
                for i in range(0, self.users, users_per_step):
                    end_index = min(i + users_per_step, self.users)
                    for j in range(i, end_index):
                        future = executor.submit(workers[j].run, self.duration)
                        futures.append(future)
                        active_workers += 1
                    
                    logger.info(f"Started {active_workers}/{self.users} users")
                    time.sleep(step_duration)
                
                # Collect results
                for future in as_completed(futures):
                    worker_results, worker_errors = future.result()
                    results.extend(worker_results)
                    errors.extend(worker_errors)
        else:
            # Start all workers at once
            with ThreadPoolExecutor(max_workers=self.users) as executor:
                futures = [executor.submit(worker.run, self.duration) for worker in workers]
                
                # Collect results
                for future in as_completed(futures):
                    worker_results, worker_errors = future.result()
                    results.extend(worker_results)
                    errors.extend(worker_errors)
        
        # Analyze and save results
        self._analyze_results(results, errors)
        
        return True
    
    def _check_health(self):
        """Check the health of the API."""
        url = f"{self.base_url}/api/health"
        
        try:
            response = requests.get(url)
            return response.status_code == 200
        except Exception:
            return False
    
    def _analyze_results(self, results, errors):
        """
        Analyze test results and save them.
        
        Args:
            results (list): List of request results.
            errors (list): List of errors.
        """
        if not results:
            logger.error("No results collected during the test.")
            return
        
        # Save raw results
        results_path = os.path.join(self.output_dir, 'raw_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        if errors:
            errors_path = os.path.join(self.output_dir, 'errors.json')
            with open(errors_path, 'w') as f:
                json.dump(errors, f)
        
        # Calculate metrics
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        
        response_times = [r['response_time'] for r in results]
        
        # Calculate summary statistics
        summary = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': success_rate,
            'total_errors': len(errors),
            'test_duration': self.duration,
            'throughput': total_requests / self.duration,
            'response_time': {
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'mean': statistics.mean(response_times) if response_times else 0,
                'median': statistics.median(response_times) if response_times else 0,
                'percentile_90': np.percentile(response_times, 90) if response_times else 0,
                'percentile_95': np.percentile(response_times, 95) if response_times else 0,
                'percentile_99': np.percentile(response_times, 99) if response_times else 0,
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        }
        
        # Add batch-specific metrics if using batch requests
        if self.batch_size > 1:
            batch_requests = [r for r in results if r['request_type'] == 'batch']
            if batch_requests:
                processing_times = [r.get('processing_time', 0) for r in batch_requests if 'processing_time' in r]
                summary['batch_requests'] = {
                    'count': len(batch_requests),
                    'total_texts_processed': sum(r.get('batch_size', 0) for r in batch_requests),
                    'processing_time': {
                        'min': min(processing_times) if processing_times else 0,
                        'max': max(processing_times) if processing_times else 0,
                        'mean': statistics.mean(processing_times) if processing_times else 0,
                        'median': statistics.median(processing_times) if processing_times else 0
                    }
                }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Log summary
        logger.info(f"Test completed with {total_requests} requests ({success_rate:.1f}% success rate)")
        logger.info(f"Throughput: {summary['throughput']:.2f} requests/second")
        logger.info(f"Response time (mean): {summary['response_time']['mean']:.4f}s")
        logger.info(f"Response time (95th percentile): {summary['response_time']['percentile_95']:.4f}s")
        
        # Generate plots
        self._generate_plots(results)
    
    def _generate_plots(self, results):
        """
        Generate visualizations of test results.
        
        Args:
            results (list): List of request results.
        """
        # Sort results by timestamp
        results.sort(key=lambda r: r['timestamp'])
        
        # Calculate timestamps relative to test start
        start_time = results[0]['timestamp']
        for r in results:
            r['relative_time'] = r['timestamp'] - start_time
        
        # Create time buckets (1-second intervals)
        buckets = defaultdict(list)
        for r in results:
            bucket = int(r['relative_time'])
            buckets[bucket].append(r)
        
        # Calculate throughput per second
        throughput_data = [(t, len(reqs)) for t, reqs in sorted(buckets.items())]
        
        # Calculate response time statistics per second
        response_time_data = []
        for t, reqs in sorted(buckets.items()):
            response_times = [r['response_time'] for r in reqs]
            if response_times:
                mean_rt = statistics.mean(response_times)
                median_rt = statistics.median(response_times)
                p95_rt = np.percentile(response_times, 95)
                response_time_data.append((t, mean_rt, median_rt, p95_rt))
        
        # Calculate success rate per second
        success_data = []
        for t, reqs in sorted(buckets.items()):
            success_count = sum(1 for r in reqs if r['success'])
            total_count = len(reqs)
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            success_data.append((t, success_rate))
        
        # Create figure with subplots
        plt.figure(figsize=(12, 18))
        
        # Plot throughput
        plt.subplot(3, 1, 1)
        times, throughputs = zip(*throughput_data) if throughput_data else ([], [])
        plt.plot(times, throughputs)
        plt.title('Throughput (Requests per Second)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Requests')
        plt.grid(True)
        
        # Plot response times
        plt.subplot(3, 1, 2)
        if response_time_data:
            times, means, medians, p95s = zip(*response_time_data)
            plt.plot(times, means, label='Mean')
            plt.plot(times, medians, label='Median')
            plt.plot(times, p95s, label='95th Percentile')
            plt.title('Response Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Response Time (seconds)')
            plt.legend()
            plt.grid(True)
        
        # Plot success rate
        plt.subplot(3, 1, 3)
        times, rates = zip(*success_data) if success_data else ([], [])
        plt.plot(times, rates)
        plt.title('Success Rate')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 105)
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.output_dir, 'load_test_plots.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Create histogram of response times
        plt.figure(figsize=(10, 6))
        response_times = [r['response_time'] for r in results]
        plt.hist(response_times, bins=50)
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Count')
        plt.grid(True)
        
        # Save histogram
        hist_path = os.path.join(self.output_dir, 'response_time_histogram.png')
        plt.savefig(hist_path)
        plt.close()

def main():
    """Main function to run the load test."""
    args = parse_args()
    
    # Validate arguments
    if args.users <= 0:
        logger.error("Number of users must be greater than 0.")
        sys.exit(1)
    
    if args.duration <= 0:
        logger.error("Duration must be greater than 0.")
        sys.exit(1)
    
    if args.batch_size <= 0:
        logger.error("Batch size must be greater than 0.")
        sys.exit(1)
    
    # Load texts
    texts = load_texts_from_file(args.file)
    if not texts:
        logger.error("No texts loaded from file.")
        sys.exit(1)
    
    # Run load test
    load_test = LoadTest(args)
    success = load_test.run()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()