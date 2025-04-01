#!/usr/bin/env python
"""
Script to evaluate the BERT sentiment analysis model.

This script loads a trained model and evaluates it on the test set,
generating comprehensive evaluation metrics and visualizations.
"""

import os
import json
import argparse
import logging
from datetime import datetime

import torch

# Add the parent directory to the path to import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_data_loaders
from src.models.model import load_model
from src.evaluation.metrics import evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis model')
    parser.add_argument('--data_dir', required=True, help='Directory containing preprocessed data')
    parser.add_argument('--model_path', required=True, help='Path to the model file')
    parser.add_argument('--output_dir', required=True, help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to evaluate on')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of sentiment classes')
    parser.add_argument('--labels', nargs='+', default=['Negative', 'Neutral', 'Positive'], help='Class labels')
    
    return parser.parse_args()

def main():
    """Main function to evaluate the model."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log arguments
    logger.info("Evaluating with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Save evaluation configuration
    config_path = os.path.join(args.output_dir, 'evaluation_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create data loaders
    logger.info(f"Creating data loaders from {args.data_dir}")
    _, _, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(
        model_path=args.model_path,
        device=args.device,
        num_classes=args.num_classes
    )
    
    # Evaluate model
    logger.info("Evaluating model")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        output_dir=args.output_dir,
        device=args.device,
        labels=args.labels
    )
    
    # Print key metrics
    logger.info("Evaluation results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    
    # Add timestamp to metrics
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save metrics (again, with timestamp)
    metrics_path = os.path.join(args.output_dir, 'metrics_with_timestamp.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()