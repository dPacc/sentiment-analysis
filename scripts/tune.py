#!/usr/bin/env python
"""
Script to perform hyperparameter tuning for the BERT sentiment analysis model.

This script trains models with different hyperparameter configurations and
selects the best configuration based on validation performance.
"""

import os
import json
import argparse
import logging
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add the parent directory to the path to import local modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_data_loaders
from src.models.training import hyperparameter_tuning
from src.evaluation.metrics import compare_models, evaluate_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Tune hyperparameters for sentiment analysis model')
    parser.add_argument('--data_dir', required=True, help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', required=True, help='Directory to save tuning results')
    parser.add_argument('--config', required=True, help='Path to hyperparameter configuration file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to train on')
    parser.add_argument('--model_name', default='bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of sentiment classes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def main():
    """Main function to tune hyperparameters."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load hyperparameter configurations
    logger.info(f"Loading hyperparameter configurations from {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    hyperparameters = config.get('hyperparameters', [])
    if not hyperparameters:
        logger.error("No hyperparameters found in configuration file")
        sys.exit(1)
    
    # Update hyperparameters with model name and num_classes
    for param_config in hyperparameters:
        param_config['model_name'] = args.model_name
        param_config['num_classes'] = args.num_classes
    
    # Log configurations
    logger.info(f"Tuning with {len(hyperparameters)} hyperparameter configurations")
    
    # Save tuning configuration
    tuning_config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'device': args.device,
        'model_name': args.model_name,
        'num_classes': args.num_classes,
        'seed': args.seed,
        'num_configurations': len(hyperparameters),
        'hyperparameters': hyperparameters,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_path = os.path.join(args.output_dir, 'tuning_config.json')
    with open(config_path, 'w') as f:
        json.dump(tuning_config, f, indent=4)
    
    # Create data loaders
    logger.info(f"Creating data loaders from {args.data_dir}")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=16,  # Default, will be overridden by hyperparameters
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed
    )
    
    # Run hyperparameter tuning
    logger.info("Starting hyperparameter tuning")
    best_model, best_hyperparams = hyperparameter_tuning(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=args.output_dir,
        hyperparameters=hyperparameters,
        device=args.device
    )
    
    # Save best hyperparameters
    best_path = os.path.join(args.output_dir, 'best_hyperparameters.json')
    with open(best_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    
    # Evaluate best model on test set
    logger.info("Evaluating best model on test set")
    best_model_dir = os.path.join(args.output_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)
    
    metrics = evaluate_model(
        model=best_model,
        test_loader=test_loader,
        output_dir=best_model_dir,
        device=args.device,
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    # Print key metrics
    logger.info("Best model evaluation results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    
    # Create comparison plots
    logger.info("Creating comparison plots")
    
    # Get results from each configuration
    results = []
    for i, _ in enumerate(hyperparameters):
        config_dir = os.path.join(args.output_dir, f"config_{i+1}")
        metrics_path = os.path.join(config_dir, 'metrics.json')
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                config_metrics = json.load(f)
            
            hyperparams_path = os.path.join(config_dir, 'hyperparameters.json')
            with open(hyperparams_path, 'r') as f:
                config_hyperparams = json.load(f)
            
            result = {
                'config_id': i + 1,
                'accuracy': config_metrics.get('accuracy', 0),
                'f1_macro': config_metrics.get('f1_macro', 0),
                'precision_macro': config_metrics.get('precision_macro', 0),
                'recall_macro': config_metrics.get('recall_macro', 0),
                'learning_rate': config_hyperparams.get('learning_rate', 0),
                'batch_size': config_hyperparams.get('batch_size', 0),
                'num_epochs': config_hyperparams.get('num_epochs', 0),
                'weight_decay': config_hyperparams.get('weight_decay', 0),
                'warmup_ratio': config_hyperparams.get('warmup_ratio', 0),
                'dropout_prob': config_hyperparams.get('dropout_prob', 0)
            }
            
            results.append(result)
    
    # Create dataframe for visualization
    df = pd.DataFrame(results)
    
    # Plot accuracy by hyperparameter
    plt.figure(figsize=(15, 20))
    
    plt.subplot(3, 2, 1)
    sns.barplot(x='learning_rate', y='accuracy', data=df)
    plt.title('Accuracy by Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    
    plt.subplot(3, 2, 2)
    sns.barplot(x='batch_size', y='accuracy', data=df)
    plt.title('Accuracy by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy')
    
    plt.subplot(3, 2, 3)
    sns.barplot(x='num_epochs', y='accuracy', data=df)
    plt.title('Accuracy by Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    
    plt.subplot(3, 2, 4)
    sns.barplot(x='weight_decay', y='accuracy', data=df)
    plt.title('Accuracy by Weight Decay')
    plt.xlabel('Weight Decay')
    plt.ylabel('Accuracy')
    
    plt.subplot(3, 2, 5)
    sns.barplot(x='warmup_ratio', y='accuracy', data=df)
    plt.title('Accuracy by Warmup Ratio')
    plt.xlabel('Warmup Ratio')
    plt.ylabel('Accuracy')
    
    plt.subplot(3, 2, 6)
    sns.barplot(x='dropout_prob', y='accuracy', data=df)
    plt.title('Accuracy by Dropout Probability')
    plt.xlabel('Dropout Probability')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.output_dir, 'hyperparameter_comparison.png')
    plt.savefig(plot_path)
    
    # Plot metrics for each configuration
    plt.figure(figsize=(12, 8))
    
    df_sorted = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    df_melted = pd.melt(df_sorted, id_vars=['config_id'], value_vars=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
    
    sns.barplot(x='config_id', y='value', hue='variable', data=df_melted)
    plt.title('Metrics by Configuration')
    plt.xlabel('Configuration ID')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    
    # Save plot
    metrics_plot_path = os.path.join(args.output_dir, 'metrics_comparison.png')
    plt.savefig(metrics_plot_path)
    
    logger.info(f"Hyperparameter tuning completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()