# Model Evaluation

This document describes the model evaluation process for the Twitter sentiment analysis project.

## Overview

The evaluation pipeline assesses the performance of the sentiment analysis model using various metrics. It provides insights into model strengths and weaknesses across different sentiment classes and generates visualizations for better interpretation.

## Evaluation Metrics

The following metrics are used to evaluate the model:

### Primary Metrics

- **Accuracy**: Proportion of correct predictions among the total number of predictions.
- **Precision**: Proportion of true positive predictions among all positive predictions.
- **Recall**: Proportion of true positive predictions among all actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

### Averaging Methods

- **Macro**: Average of the per-class metrics, giving equal weight to each class.
- **Weighted**: Average of the per-class metrics, weighted by the number of samples in each class.

### Additional Analysis

- **Confusion Matrix**: Table showing true positives, false positives, true negatives, and false negatives for each class.
- **Per-Class Metrics**: Precision, recall, and F1 score for each sentiment class.
- **Classification Report**: Detailed report with per-class and overall metrics.

## Implementation

The evaluation pipeline is implemented in the `src/evaluation/metrics.py` module. Below are the key functions:

### `calculate_metrics(y_true, y_pred, labels=None)`

Calculates various evaluation metrics.

**Parameters**:
- `y_true` (array-like): Ground truth labels.
- `y_pred` (array-like): Predicted labels.
- `labels` (list, optional): List of label names.

**Returns**:
- dict: Dictionary containing various metrics.

### `plot_confusion_matrix(y_true, y_pred, labels=None, output_path=None)`

Plots confusion matrix.

**Parameters**:
- `y_true` (array-like): Ground truth labels.
- `y_pred` (array-like): Predicted labels.
- `labels` (list, optional): List of label names.
- `output_path` (str, optional): Path to save the plot.

**Returns**:
- matplotlib.figure.Figure: Confusion matrix figure.

### `plot_metrics_over_epochs(history, output_path=None)`

Plots metrics over epochs.

**Parameters**:
- `history` (dict): Training history with metrics over epochs.
- `output_path` (str, optional): Path to save the plot.

**Returns**:
- matplotlib.figure.Figure: Metrics plot figure.

### `evaluate_model(model, test_loader, output_dir, device='cuda', labels=None)`

Evaluates the model and generates a comprehensive evaluation report.

**Parameters**:
- `model` (BertSentimentClassifier): Model to evaluate.
- `test_loader` (DataLoader): Test data loader.
- `output_dir` (str): Directory to save evaluation results.
- `device` (str): Device to evaluate on ('cuda' or 'cpu').
- `labels` (list, optional): List of label names.

**Returns**:
- dict: Dictionary containing evaluation metrics.

### `compare_models(model_dirs, output_dir, metric='accuracy')`

Compares multiple models based on their evaluation metrics.

**Parameters**:
- `model_dirs` (list): List of directories containing model evaluation results.
- `output_dir` (str): Directory to save comparison results.
- `metric` (str): Metric to use for comparison.

**Returns**:
- dict: Dictionary containing comparison results.

## Visualizations

The evaluation pipeline generates the following visualizations:

### Confusion Matrix

A heatmap showing the number of samples for each combination of true and predicted sentiment classes. This helps identify which classes the model confuses most often.

### Metrics Over Epochs

Line plots showing the training and validation loss, as well as validation accuracy, over the training epochs. This helps identify potential overfitting or underfitting.

### Model Comparison

Bar chart comparing multiple models based on a specified metric (e.g., accuracy). This helps identify the best-performing model among different configurations.

## Usage

To evaluate a model, run the following command:

```bash
python -m scripts.evaluate --data_dir ./processed_data --model_path ./models/best_model.pt --output_dir ./evaluation
```

To compare multiple models, use the following command:

```bash
python -m scripts.compare_models --model_dirs ./model1 ./model2 ./model3 --output_dir ./comparison --metric f1_macro
```

## Expected Output

The evaluation pipeline generates the following output files in the specified output directory:

- `metrics.json`: JSON file containing all calculated metrics.
- `confusion_matrix.png`: Visualization of the confusion matrix.
- `metrics_plot.png`: Plot of metrics over epochs (if training history is available).
- `model_comparison.png`: Bar chart comparing multiple models (if using compare_models).

## Example Evaluation Results

Below is an example of evaluation results for a model:

```json
{
    "accuracy": 0.8712,
    "precision_macro": 0.8523,
    "recall_macro": 0.8645,
    "f1_macro": 0.8583,
    "precision_weighted": 0.8734,
    "recall_weighted": 0.8712,
    "f1_weighted": 0.8721,
    "confusion_matrix": [
        [423, 52, 25],
        [38, 521, 41],
        [19, 36, 645]
    ],
    "classification_report": {
        "Negative": {
            "precision": 0.8833,
            "recall": 0.8460,
            "f1-score": 0.8642,
            "support": 500
        },
        "Neutral": {
            "precision": 0.8557,
            "recall": 0.8683,
            "f1-score": 0.8620,
            "support": 600
        },
        "Positive": {
            "precision": 0.9077,
            "recall": 0.9200,
            "f1-score": 0.9138,
            "support": 700
        },
        "accuracy": 0.8712,
        "macro avg": {
            "precision": 0.8823,
            "recall": 0.8781,
            "f1-score": 0.8800,
            "support": 1800
        },
        "weighted avg": {
            "precision": 0.8834,
            "recall": 0.8812,
            "f1-score": 0.8821,
            "support": 1800
        }
    },
    "precision_Negative": 0.8833,
    "recall_Negative": 0.8460,
    "f1_Negative": 0.8642,
    "precision_Neutral": 0.8557,
    "recall_Neutral": 0.8683,
    "f1_Neutral": 0.8620,
    "precision_Positive": 0.9077,
    "recall_Positive": 0.9200,
    "f1_Positive": 0.9138
}
```

## Interpreting Results

When interpreting evaluation results, consider the following:

- **Class Imbalance**: If the dataset has imbalanced classes, weighted metrics give a better overall picture.
- **Per-Class Performance**: Check if the model performs consistently across all sentiment classes.
- **Confusion Matrix**: Identify which sentiment classes the model frequently confuses.
- **F1 Score**: Use F1 score as a balanced metric that considers both precision and recall.
- **Training Curves**: Look for signs of overfitting (validation loss increases while training loss decreases) or underfitting (both losses remain high).

## Best Practices

- Always evaluate on a held-out test set that was not used during training or validation.
- Use multiple metrics to get a comprehensive view of model performance.
- Compare different model configurations to identify the best approach.
- Consider the specific requirements of your application when interpreting results (e.g., whether precision or recall is more important).
- Analyze errors qualitatively to understand model limitations and guide future improvements.