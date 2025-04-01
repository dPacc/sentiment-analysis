"""
Model module for Twitter sentiment analysis.

This module defines the BERT-based sentiment analysis model.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BertSentimentClassifier(nn.Module):
    """
    BERT-based sentiment classifier model.
    
    This model fine-tunes a pre-trained BERT model for sentiment analysis,
    with three sentiment classes: negative (0), neutral (1), and positive (2).
    """
    
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=3, dropout_prob=0.1):
        """
        Initialize the BERT sentiment classifier.
        
        Args:
            bert_model_name (str): Name of the pre-trained BERT model.
            num_classes (int): Number of sentiment classes.
            dropout_prob (float): Dropout probability for the classifier head.
        """
        super(BertSentimentClassifier, self).__init__()
        
        # Log model initialization
        logger.info(f"Initializing BertSentimentClassifier with {bert_model_name}")
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Get BERT configuration
        config = self.bert.config
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        # Initialize weights for the classifier layer
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
        logger.info(f"Model initialized with {num_classes} output classes")
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type IDs.
            labels (torch.Tensor): Ground truth labels for loss computation.
            
        Returns:
            tuple: Tuple containing loss (if labels are provided) and logits.
        """
        # Create input dictionary for BERT
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Add token_type_ids if provided
        if token_type_ids is not None:
            inputs['token_type_ids'] = token_type_ids
        
        # Get BERT outputs
        outputs = self.bert(**inputs)
        
        # Take the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Compute logits
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        return (loss, logits) if loss is not None else logits

def load_model(model_path, device='cuda', num_classes=3):
    """
    Load a saved model.
    
    Args:
        model_path (str): Path to the saved model file.
        device (str): Device to load the model on ('cuda' or 'cpu').
        num_classes (int): Number of sentiment classes.
        
    Returns:
        BertSentimentClassifier: Loaded model.
    """
    logger.info(f"Loading model from {model_path} to device {device}")
    
    try:
        # Initialize model
        model = BertSentimentClassifier(num_classes=num_classes)
        
        # Load state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Move model to device
        model = model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info(f"Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def save_model(model, model_path):
    """
    Save model to disk.
    
    Args:
        model (BertSentimentClassifier): Model to save.
        model_path (str): Path to save the model.
    """
    logger.info(f"Saving model to {model_path}")
    
    try:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state dictionary
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    model = BertSentimentClassifier()
    print(model)
    
    # Test forward pass
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    labels = torch.randint(0, 3, (batch_size,))
    
    loss, logits = model(input_ids, attention_mask, labels=labels)
    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")