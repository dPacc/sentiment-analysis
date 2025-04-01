# Data Preprocessing

This document describes the data preprocessing pipeline for the Twitter sentiment analysis project.

## Overview

The preprocessing pipeline transforms raw Twitter data into a clean, tokenized format suitable for training a BERT-based sentiment analysis model. The pipeline handles various aspects of Twitter text, including emoji, mentions, hashtags, and special characters.

## Pipeline Steps

The preprocessing pipeline consists of the following steps:

1. **Data Loading**: Load the raw data from CSV format.
2. **Text Cleaning**:
   - Convert text to lowercase
   - Remove URLs
   - Remove mentions and hashtags
   - Remove emojis and special characters
   - Remove extra whitespace
   - Optionally remove stopwords
3. **Tokenization**: Tokenize the cleaned text using the BERT tokenizer.
4. **Saving**: Save the preprocessed data for later use.

## Implementation

The preprocessing pipeline is implemented in the `src/data/preprocessing.py` module. Below are the key functions:

### `load_data(filepath)`

Loads the dataset from a CSV file.

**Parameters**:
- `filepath` (str): Path to the dataset CSV file.

**Returns**:
- pandas.DataFrame: Loaded dataset.

### `clean_text(text, remove_stopwords=True)`

Cleans the text data by removing URLs, mentions, hashtags, emojis, and special characters.

**Parameters**:
- `text` (str): Text to clean.
- `remove_stopwords` (bool): Whether to remove stopwords.

**Returns**:
- str: Cleaned text.

### `tokenize_data(texts, max_length=128)`

Tokenizes the text data using BERT tokenizer.

**Parameters**:
- `texts` (list): List of texts to tokenize.
- `max_length` (int): Maximum sequence length.

**Returns**:
- dict: Dictionary containing input_ids, attention_mask, and token_type_ids.

### `preprocess_dataset(input_filepath, output_dir, config=None)`

Main function to preprocess the dataset.

**Parameters**:
- `input_filepath` (str): Path to the input dataset.
- `output_dir` (str): Directory to save preprocessed data.
- `config` (dict): Configuration parameters for preprocessing.

**Returns**:
- tuple: Tuple containing the preprocessed data and tokenized inputs.

## Configuration

The preprocessing pipeline can be configured using a JSON configuration file. Below is an example configuration:

```json
{
    "remove_stopwords": true,
    "max_length": 128,
    "train_test_split": 0.2,
    "random_state": 42
}
```

## Usage

To use the preprocessing pipeline, run the following command:

```bash
python -m src.data.preprocessing --input path/to/dataset.csv --output ./processed_data --config ./config/preprocessing_config.json
```

## Expected Output

The preprocessing pipeline generates the following output files in the specified output directory:

- `preprocessed_data.csv`: The cleaned dataset with an additional `cleaned_text` column.
- `tokenized_data.json`: The tokenized data ready for model training.

## Example

Here's an example of the preprocessing pipeline in action:

### Raw Text:
```
@user I just love this product! It's amazing :) #happy #customer https://example.com
```

### Cleaned Text (with stopwords removed):
```
love product amazing happy customer
```

### Tokenized Output:
```
{
    "input_ids": [101, 2293, 2135, 12662, 4062, 102, 0, 0, ...],
    "attention_mask": [1, 1, 1, 1, 1, 1, 0, 0, ...],
    "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, ...]
}
```

## Notes

- The preprocessing pipeline is designed to handle Twitter-specific features like mentions, hashtags, and emojis.
- NLTK is used for stopword removal, which requires downloading the necessary resources.
- The BERT tokenizer is used for tokenization, which handles out-of-vocabulary words using WordPiece subword tokenization.
- The maximum sequence length is set to 128 by default, which is appropriate for most Twitter data.