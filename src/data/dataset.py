"""
Dataset module for Twitter sentiment analysis.

This module creates PyTorch datasets for training and evaluation of the sentiment analysis model.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwitterSentimentDataset(Dataset):
    """
    PyTorch Dataset for Twitter sentiment analysis.
    
    This dataset loads preprocessed and tokenized data for training and evaluation.
    """
    
    def __init__(self, data_df, tokenized_data=None, tokenizer=None, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            data_df (pandas.DataFrame): DataFrame containing the preprocessed data.
            tokenized_data (dict, optional): Pre-tokenized data. If None, will tokenize using tokenizer.
            tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer to use if tokenized_data is None.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.data = data_df
        
        # Map sentiment labels to integers
        sentiment_mapping = {
            'positive': 2,
            'neutral': 1,
            'negative': 0
        }
        
        # Convert string labels to integers
        self.labels = self.data['sentiment'].map(lambda x: sentiment_mapping.get(x.lower(), 1)).values
        
        # Use pre-tokenized data if available
        if tokenized_data is not None:
            self.encodings = tokenized_data
        # Otherwise, tokenize using the provided tokenizer
        elif tokenizer is not None:
            texts = self.data['cleaned_text'].tolist()
            self.encodings = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
        else:
            raise ValueError("Either tokenized_data or tokenizer must be provided.")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, token_type_ids, and labels.
        """
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in self.encodings:
            item['token_type_ids'] = self.encodings['token_type_ids'][idx]
            
        item['labels'] = torch.tensor(self.labels[idx])
        
        return item

def load_tokenized_data(filepath):
    """
    Load tokenized data from a file.
    
    Args:
        filepath (str): Path to the tokenized data file.
        
    Returns:
        dict: Dictionary containing the tokenized data.
    """
    logger.info(f"Loading tokenized data from {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to PyTorch tensors
        tokenized_data = {k: torch.tensor(v) for k, v in data.items()}
        
        logger.info(f"Tokenized data loaded successfully.")
        return tokenized_data
    except Exception as e:
        logger.error(f"Error loading tokenized data: {e}")
        raise

def create_data_loaders(data_dir, batch_size=16, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir (str): Directory containing preprocessed and tokenized data.
        batch_size (int): Batch size for data loaders.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: Tuple containing train_loader, val_loader, and test_loader.
    """
    logger.info(f"Creating data loaders with batch_size={batch_size}")
    
    # Load preprocessed data
    data_path = os.path.join(data_dir, 'preprocessed_data.csv')
    data_df = pd.read_csv(data_path)
    
    # Load tokenized data
    tokenized_path = os.path.join(data_dir, 'tokenized_data.json')
    tokenized_data = load_tokenized_data(tokenized_path)
    
    # Create dataset
    full_dataset = TwitterSentimentDataset(data_df, tokenized_data)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split dataset
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Data loaders created successfully.")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    import argparse
    from transformers import BertTokenizer
    
    parser = argparse.ArgumentParser(description='Create data loaders for sentiment analysis')
    parser.add_argument('--data_dir', required=True, help='Directory containing preprocessed data')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Display sample batch
    for batch in train_loader:
        print(f"Batch size: {batch['input_ids'].shape}")
        print(f"Labels: {batch['labels']}")
        break