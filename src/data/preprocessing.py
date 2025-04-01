"""
Data preprocessing module for Twitter sentiment analysis.

This module handles the preprocessing of the Twitter Entity Sentiment Analysis Dataset,
including loading, cleaning, and tokenization of the data.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import logging
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
def download_nltk_resources():
    """Download necessary NLTK resources if they're not already downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        logger.info("Downloading necessary NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        logger.info("NLTK resources downloaded successfully.")
        
def load_data(filepath):
    """
    Load the Twitter sentiment analysis dataset.
    
    Args:
        filepath (str): Path to the dataset CSV file.
        
    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_text(text, remove_stopwords=True):
    """
    Clean the text data by removing URLs, mentions, hashtags, emojis, 
    and special characters. Optionally remove stopwords.
    
    Args:
        text (str): Text to clean.
        remove_stopwords (bool): Whether to remove stopwords.
        
    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords if specified
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text = ' '.join([word for word in word_tokens if word not in stop_words])
    
    return text

def tokenize_data(texts, max_length=128):
    """
    Tokenize the text data using BERT tokenizer.
    
    Args:
        texts (list): List of texts to tokenize.
        max_length (int): Maximum sequence length.
        
    Returns:
        dict: Dictionary containing input_ids, attention_mask, and token_type_ids.
    """
    logger.info(f"Tokenizing {len(texts)} texts with max_length={max_length}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    encoded_inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    logger.info(f"Tokenization complete. Sample shape: {encoded_inputs['input_ids'].shape}")
    return encoded_inputs

def save_preprocessed_data(data, output_dir, filename):
    """
    Save preprocessed data to disk.
    
    Args:
        data (dict or DataFrame): Preprocessed data to save.
        output_dir (str): Directory to save the data.
        filename (str): Name of the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    logger.info(f"Saving preprocessed data to {output_path}")
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
    else:
        # For tokenized data, save in a suitable format
        # This is a simplified approach - in production, consider more efficient formats
        with open(output_path, 'w') as f:
            json.dump({k: v.tolist() for k, v in data.items()}, f)
    
    logger.info(f"Data saved successfully to {output_path}")

def preprocess_dataset(input_filepath, output_dir, config=None):
    """
    Main function to preprocess the dataset.
    
    Args:
        input_filepath (str): Path to the input dataset.
        output_dir (str): Directory to save preprocessed data.
        config (dict): Configuration parameters for preprocessing.
        
    Returns:
        tuple: Tuple containing the preprocessed data and tokenized inputs.
    """
    # Set default configuration if not provided
    if config is None:
        config = {
            'remove_stopwords': True,
            'max_length': 128,
            'train_test_split': 0.2,
            'random_state': 42
        }
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load data
    df = load_data(input_filepath)
    
    # Display dataset info
    logger.info(f"Dataset columns: {df.columns.tolist()}")
    logger.info(f"Dataset info:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isna().sum()}")
    
    # Check for sentiment labels
    if 'sentiment' not in df.columns:
        logger.error("Dataset does not contain 'sentiment' column")
        raise ValueError("Dataset must contain a 'sentiment' column")
    
    # Check for text column
    text_col = 'text' if 'text' in df.columns else 'tweet'
    if text_col not in df.columns:
        logger.error(f"Dataset does not contain '{text_col}' column")
        raise ValueError(f"Dataset must contain a '{text_col}' column")
    
    # Clean text data
    logger.info(f"Cleaning text data with remove_stopwords={config['remove_stopwords']}")
    df['cleaned_text'] = df[text_col].apply(lambda x: clean_text(x, config['remove_stopwords']))
    
    # Handle missing values
    df = df.dropna(subset=['cleaned_text', 'sentiment'])
    logger.info(f"Dataset after cleaning: {df.shape}")
    
    # Tokenize data
    tokenized_inputs = tokenize_data(df['cleaned_text'].tolist(), config['max_length'])
    
    # Save preprocessed data
    save_preprocessed_data(df, output_dir, 'preprocessed_data.csv')
    save_preprocessed_data(tokenized_inputs, output_dir, 'tokenized_data.json')
    
    return df, tokenized_inputs

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Twitter sentiment analysis dataset')
    parser.add_argument('--input', required=True, help='Path to input dataset')
    parser.add_argument('--output', default='./processed_data', help='Output directory')
    parser.add_argument('--config', default=None, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Run preprocessing
    preprocess_dataset(args.input, args.output, config)