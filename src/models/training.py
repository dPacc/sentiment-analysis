"""
Training module for Twitter sentiment analysis.

This module handles training and hyperparameter tuning for the BERT sentiment analysis model.
"""

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import logging
import os
import json
from datetime import datetime

# Local imports
from ..models.model import BertSentimentClassifier, save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """
    Train the model for one epoch.
    
    Args:
        model (BertSentimentClassifier): Model to train.
        data_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer for parameter updates.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        
    Returns:
        float: Average training loss for the epoch.
    """
    # Set model to training mode
    model.train()
    
    # Initialize tracking variables
    total_loss = 0
    num_batches = len(data_loader)
    
    # Training loop
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss, _ = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids'),
            labels=batch['labels']
        )
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update tracking variables
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    
    logger.info(f"Training completed with average loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, data_loader, device):
    """
    Evaluate the model.
    
    Args:
        model (BertSentimentClassifier): Model to evaluate.
        data_loader (DataLoader): Evaluation data loader.
        device (torch.device): Device to evaluate on.
        
    Returns:
        tuple: Tuple containing average loss, accuracy, and predictions.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize tracking variables
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            loss, logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch.get('token_type_ids'),
                labels=batch['labels']
            )
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Update tracking variables
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    logger.info(f"Evaluation completed with average loss: {avg_loss:.4f}, accuracy: {accuracy:.4f}")
    return avg_loss, accuracy, (all_preds, all_labels)

def train_model(
    train_loader, 
    val_loader, 
    model_save_dir,
    num_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    device='cuda',
    model_name='bert-base-uncased',
    num_classes=3
):
    """
    Train the BERT sentiment classifier.
    
    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        model_save_dir (str): Directory to save model checkpoints.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        warmup_ratio (float): Ratio of training steps for learning rate warmup.
        device (str): Device to train on ('cuda' or 'cpu').
        model_name (str): Name of the pre-trained BERT model.
        num_classes (int): Number of sentiment classes.
        
    Returns:
        tuple: Tuple containing the trained model and training history.
    """
    # Create device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Initialize model
    model = BertSentimentClassifier(
        bert_model_name=model_name,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Create optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Initialize best validation accuracy
    best_val_accuracy = 0.0
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Evaluate model
        val_loss, val_accuracy, _ = evaluate(model, val_loader, device)
        
        # Update training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Save model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join(model_save_dir, 'best_model.pt')
            save_model(model, model_path)
            logger.info(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        save_model(model, checkpoint_path)
        
        logger.info(f"Epoch {epoch+1} completed: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
    
    # Save training history
    history_path = os.path.join(model_save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    return model, history

def hyperparameter_tuning(
    train_loader,
    val_loader,
    test_loader,
    output_dir,
    hyperparameters,
    device='cuda'
):
    """
    Tune hyperparameters by training models with different configurations.
    
    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        output_dir (str): Directory to save model checkpoints and results.
        hyperparameters (dict): Dictionary of hyperparameter configurations to try.
        device (str): Device to train on ('cuda' or 'cpu').
        
    Returns:
        tuple: Tuple containing the best model and best hyperparameters.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize best validation accuracy
    best_val_accuracy = 0.0
    best_hyperparams = None
    best_model = None
    
    # Initialize results dictionary
    results = {
        'configurations': [],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Hyperparameter tuning loop
    logger.info(f"Starting hyperparameter tuning with {len(hyperparameters)} configurations")
    for i, params in enumerate(hyperparameters):
        logger.info(f"Training configuration {i+1}/{len(hyperparameters)}: {params}")
        
        # Create model save directory
        model_save_dir = os.path.join(output_dir, f"config_{i+1}")
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Save hyperparameters
        with open(os.path.join(model_save_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(params, f)
        
        # Train model
        model, history = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model_save_dir=model_save_dir,
            **params,
            device=device
        )
        
        # Evaluate on test set
        test_loss, test_accuracy, _ = evaluate(model, test_loader, device)
        
        # Save results
        result = {
            'hyperparameters': params,
            'val_accuracy': history['val_accuracy'][-1],
            'test_accuracy': test_accuracy,
            'test_loss': test_loss
        }
        results['configurations'].append(result)
        
        # Update best model
        if history['val_accuracy'][-1] > best_val_accuracy:
            best_val_accuracy = history['val_accuracy'][-1]
            best_hyperparams = params
            best_model = model
            logger.info(f"New best configuration with validation accuracy: {best_val_accuracy:.4f}")
        
        # Save results after each configuration
        results_path = os.path.join(output_dir, 'tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
    
    logger.info(f"Hyperparameter tuning completed. Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Best hyperparameters: {best_hyperparams}")
    
    return best_model, best_hyperparams

if __name__ == "__main__":
    # Example usage
    import argparse
    from ..data.dataset import create_data_loaders
    
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data_dir', required=True, help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', required=True, help='Directory to save model and results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Train model
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model_save_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    # Evaluate on test set
    test_loss, test_accuracy, _ = evaluate(model, test_loader, 'cuda')
    print(f"Test accuracy: {test_accuracy:.4f}")