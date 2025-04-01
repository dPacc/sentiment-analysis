#!/usr/bin/env python
"""
Script to train the BERT sentiment analysis model.

This script loads preprocessed data, creates data loaders, initializes the model,
and trains it with the specified hyperparameters.
"""

import os
import json
import argparse
import logging
from datetime import datetime

import torch
import matplotlib.pyplot as plt

# Add the parent directory to the path to import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_data_loaders
from src.models.model import BertSentimentClassifier
from src.models.training import train_model, evaluate
from src.evaluation.metrics import plot_confusion_matrix, plot_metrics_over_epochs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data_dir', required=True, help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', required=True, help='Directory to save model and results')
    parser.add_argument('--config', default=None, help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to train on')
    parser.add_argument('--model_name', default='bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of sentiment classes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def main():
    """Main function to train the model."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Log arguments
    logger.info("Training with the following parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create data loaders
    logger.info(f"Creating data loaders from {args.data_dir}")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed
    )
    
    # Train model
    logger.info("Starting model training")
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model_save_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        device=args.device,
        model_name=args.model_name,
        num_classes=args.num_classes
    )
    
    # Plot training history
    logger.info("Plotting training history")
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_metrics_over_epochs(history, history_plot_path)
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_loss, test_accuracy, (test_preds, test_labels) = evaluate(model, test_loader, args.device)
    logger.info(f"Test results: loss={test_loss:.4f}, accuracy={test_accuracy:.4f}")
    
    # Plot confusion matrix
    logger.info("Plotting confusion matrix")
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, ['Negative', 'Neutral', 'Positive'], cm_path)
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    logger.info(f"Training completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()