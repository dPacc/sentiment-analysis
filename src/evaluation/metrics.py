"""
Metrics module for Twitter sentiment analysis.

This module implements evaluation metrics for the sentiment analysis model.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, labels=None):
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of label names.
        
    Returns:
        dict: Dictionary containing various metrics.
    """
    # Default label names if not provided
    if labels is None:
        labels = ['Negative', 'Neutral', 'Positive']
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    }
    
    # Add per-class metrics
    for i, label in enumerate(labels):
        metrics[f'precision_{label}'] = precision_score(y_true, y_pred, average=None)[i]
        metrics[f'recall_{label}'] = recall_score(y_true, y_pred, average=None)[i]
        metrics[f'f1_{label}'] = f1_score(y_true, y_pred, average=None)[i]
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of label names.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix figure.
    """
    # Default label names if not provided
    if labels is None:
        labels = ['Negative', 'Neutral', 'Positive']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {output_path}")
    
    return plt.gcf()

def plot_metrics_over_epochs(history, output_path=None):
    """
    Plot metrics over epochs.
    
    Args:
        history (dict): Training history with metrics over epochs.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        matplotlib.figure.Figure: Metrics plot figure.
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"Metrics plot saved to {output_path}")
    
    return plt.gcf()

def evaluate_model(model, test_loader, output_dir, device='cuda', labels=None):
    """
    Evaluate the model and generate comprehensive evaluation report.
    
    Args:
        model (BertSentimentClassifier): Model to evaluate.
        test_loader (DataLoader): Test data loader.
        output_dir (str): Directory to save evaluation results.
        device (str): Device to evaluate on ('cuda' or 'cpu').
        labels (list, optional): List of label names.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Default label names if not provided
    if labels is None:
        labels = ['Negative', 'Neutral', 'Positive']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Initialize lists for predictions and ground truth
    all_preds = []
    all_labels = []
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    import torch
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids')
            )
            
            # Get predictions
            if isinstance(outputs, tuple):
                logits = outputs[1]
            else:
                logits = outputs
                
            preds = torch.argmax(logits, dim=1)
            
            # Update lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, labels)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, labels, cm_path)
    
    # Log overall metrics
    logger.info(f"Evaluation completed with accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
    logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
    
    return metrics

def compare_models(model_dirs, output_dir, metric='accuracy'):
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        model_dirs (list): List of directories containing model evaluation results.
        output_dir (str): Directory to save comparison results.
        metric (str): Metric to use for comparison.
        
    Returns:
        dict: Dictionary containing comparison results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    results = {
        'models': [],
        'best_model': None,
        'comparison_metric': metric,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Load metrics for each model
    best_value = 0
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        metrics_path = os.path.join(model_dir, 'metrics.json')
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Check if the specified metric exists
            if metric not in metrics:
                logger.warning(f"Metric '{metric}' not found for model '{model_name}'. Using accuracy instead.")
                metric_value = metrics.get('accuracy', 0)
            else:
                metric_value = metrics[metric]
            
            # Save model results
            model_result = {
                'name': model_name,
                'metrics': metrics,
                metric: metric_value
            }
            results['models'].append(model_result)
            
            # Update best model
            if metric_value > best_value:
                best_value = metric_value
                results['best_model'] = model_name
        except Exception as e:
            logger.error(f"Error loading metrics for model '{model_name}': {e}")
    
    # Sort models by the comparison metric
    results['models'].sort(key=lambda x: x[metric], reverse=True)
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    model_names = [model['name'] for model in results['models']]
    metric_values = [model[metric] for model in results['models']]
    
    plt.bar(model_names, metric_values)
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison by {metric.capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, bbox_inches='tight')
    
    logger.info(f"Model comparison completed. Best model: {results['best_model']} with {metric}={best_value:.4f}")
    return results

if __name__ == "__main__":
    # Example usage
    import argparse
    import torch
    from ..data.dataset import create_data_loaders
    from ..models.model import load_model
    
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis model')
    parser.add_argument('--data_dir', required=True, help='Directory containing preprocessed data')
    parser.add_argument('--model_path', required=True, help='Path to the model file')
    parser.add_argument('--output_dir', required=True, help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Create data loaders
    _, _, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Load model
    model = load_model(args.model_path)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, args.output_dir)
    print(f"Accuracy: {metrics['accuracy']:.4f}")