# Model Training

This document describes the model training pipeline for the Twitter sentiment analysis project.

## Overview

The training pipeline fine-tunes a pre-trained BERT model for sentiment analysis on Twitter data. The pipeline includes dataset creation, model initialization, training loop, and hyperparameter tuning.

## Model Architecture

The sentiment analysis model is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture with a classification head on top. 

### Components:

1. **BERT Encoder**: Pre-trained BERT model (`bert-base-uncased`) that converts input tokens into contextual embeddings.
2. **Dropout Layer**: Prevents overfitting during fine-tuning.
3. **Classification Head**: Linear layer that maps the [CLS] token representation to sentiment classes.

### Output Classes:

- 0: Negative
- 1: Neutral
- 2: Positive

## Training Pipeline

The training pipeline consists of the following steps:

1. **Dataset Creation**: Create PyTorch datasets and data loaders for training, validation, and testing.
2. **Model Initialization**: Initialize the BERT model with a classification head.
3. **Training Loop**: Fine-tune the model on the training dataset.
4. **Validation**: Evaluate the model on the validation dataset.
5. **Hyperparameter Tuning**: Find optimal hyperparameters for the model.

## Implementation

The training pipeline is implemented in the `src/models/training.py` module. Below are the key functions:

### `train_epoch(model, data_loader, optimizer, scheduler, device)`

Trains the model for one epoch.

**Parameters**:
- `model` (BertSentimentClassifier): Model to train.
- `data_loader` (DataLoader): Training data loader.
- `optimizer` (Optimizer): Optimizer for parameter updates.
- `scheduler` (LRScheduler): Learning rate scheduler.
- `device` (torch.device): Device to train on.

**Returns**:
- float: Average training loss for the epoch.

### `evaluate(model, data_loader, device)`

Evaluates the model.

**Parameters**:
- `model` (BertSentimentClassifier): Model to evaluate.
- `data_loader` (DataLoader): Evaluation data loader.
- `device` (torch.device): Device to evaluate on.

**Returns**:
- tuple: Tuple containing average loss, accuracy, and predictions.

### `train_model(train_loader, val_loader, model_save_dir, ...)`

Trains the BERT sentiment classifier.

**Parameters**:
- `train_loader` (DataLoader): Training data loader.
- `val_loader` (DataLoader): Validation data loader.
- `model_save_dir` (str): Directory to save model checkpoints.
- `num_epochs` (int): Number of training epochs.
- `learning_rate` (float): Learning rate for optimizer.
- `weight_decay` (float): Weight decay for optimizer.
- `warmup_ratio` (float): Ratio of training steps for learning rate warmup.
- `device` (str): Device to train on ('cuda' or 'cpu').
- `model_name` (str): Name of the pre-trained BERT model.
- `num_classes` (int): Number of sentiment classes.

**Returns**:
- tuple: Tuple containing the trained model and training history.

### `hyperparameter_tuning(train_loader, val_loader, test_loader, output_dir, hyperparameters, device)`

Tunes hyperparameters by training models with different configurations.

**Parameters**:
- `train_loader` (DataLoader): Training data loader.
- `val_loader` (DataLoader): Validation data loader.
- `test_loader` (DataLoader): Test data loader.
- `output_dir` (str): Directory to save model checkpoints and results.
- `hyperparameters` (dict): Dictionary of hyperparameter configurations to try.
- `device` (str): Device to train on ('cuda' or 'cpu').

**Returns**:
- tuple: Tuple containing the best model and best hyperparameters.

## Hyperparameters

The following hyperparameters can be tuned:

- **Learning Rate**: Rate at which model parameters are updated.
- **Number of Epochs**: Number of complete passes through the training dataset.
- **Batch Size**: Number of samples processed before the model parameters are updated.
- **Weight Decay**: L2 regularization term to prevent overfitting.
- **Warmup Ratio**: Percentage of training steps for learning rate warmup.
- **Dropout Probability**: Probability of dropping a neuron during training.

## Recommended Hyperparameters

Based on empirical testing, the following hyperparameters work well for most Twitter sentiment analysis tasks:

```json
{
    "num_epochs": 4,
    "learning_rate": 2e-5,
    "batch_size": 16,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout_prob": 0.1
}
```

## Usage

To train the model, run the following command:

```bash
python -m scripts.train --data_dir ./processed_data --output_dir ./models --config ./config/model_config.json
```

To perform hyperparameter tuning, use the following command:

```bash
python -m scripts.tune --data_dir ./processed_data --output_dir ./tuning_results --config ./config/hyperparameter_config.json
```

## Expected Output

The training pipeline generates the following output files in the specified output directory:

- `best_model.pt`: The best model based on validation accuracy.
- `checkpoint_epoch_X.pt`: Checkpoint after each epoch.
- `training_history.json`: Training metrics history.

For hyperparameter tuning, the following additional files are generated:

- `tuning_results.json`: Results of hyperparameter tuning.
- `config_X/`: Directory for each hyperparameter configuration.

## Performance Considerations

- Training on GPU is highly recommended for reasonable training times.
- Batch size should be adjusted based on available GPU memory.
- The learning rate should be kept small (1e-5 to 5e-5) for stable training.
- Gradient clipping is used to prevent exploding gradients.
- Using a learning rate scheduler with warmup improves training stability.

## Example Training History

Below is an example of training history for a model trained with the recommended hyperparameters:

```json
{
    "train_loss": [0.4256, 0.2134, 0.1523, 0.1178],
    "val_loss": [0.2345, 0.1876, 0.1654, 0.1543],
    "val_accuracy": [0.8234, 0.8543, 0.8654, 0.8712]
}
```