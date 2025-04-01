#!/usr/bin/env python
"""
Script to run the Flask API for sentiment analysis.

This script loads a trained model and starts the Flask API server
for serving sentiment analysis predictions.
"""

import os
import argparse
import logging

# Add the parent directory to the path to import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import start_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run sentiment analysis API')
    parser.add_argument('--model_path', required=True, help='Path to the model file')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to run the model on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    return parser.parse_args()

def main():
    """Main function to run the API server."""
    args = parse_args()
    
    # Log arguments
    logger.info("Starting API server with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Check if model file exists
    if not os.path.isfile(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Start server
    try:
        logger.info(f"Starting server on {args.host}:{args.port}")
        start_server(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            debug=args.debug,
            device_name=args.device
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()