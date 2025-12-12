"""
Classification Evaluation Module
================================
Functions for evaluating classification model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score
)


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', figsize=(8, 6)):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Converted', 'Converted'],
        yticklabels=['Not Converted', 'Converted'],
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    return fig


def plot_roc_curve(y_true, y_proba, title='ROC Curve', figsize=(8, 6)):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='#0066CC', lw=2,
            label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--',
            label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#0066CC')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return fig


def plot_precision_recall_curve(y_true, y_proba, title='Precision-Recall Curve',
                                 figsize=(8, 6)):
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color='#28A745', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def get_classification_report_df(y_true, y_pred):
    """
    Get classification report as DataFrame.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        pd.DataFrame: Classification report
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    return df.round(3)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate all evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

    return metrics


def check_success_criteria(metrics, criteria):
    """
    Check if metrics meet success criteria.

    Args:
        metrics: Dictionary of achieved metrics
        criteria: Dictionary of target values

    Returns:
        tuple: (all_met, results_dict)
    """
    results = {}
    all_met = True

    for metric, target in criteria.items():
        achieved = metrics.get(metric, 0)
        passed = achieved >= target
        if not passed:
            all_met = False
        results[metric] = {
            'target': target,
            'achieved': achieved,
            'passed': passed
        }

    return all_met, results
