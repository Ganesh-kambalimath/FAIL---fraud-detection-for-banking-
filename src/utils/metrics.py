"""
Utility functions for metrics calculation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute comprehensive fraud detection metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # ROC AUC
    try:
        auc_roc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_roc = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # False positive rate and false negative rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'specificity': specificity
    }
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:   {metrics['true_positives']}")
    print(f"  True Negatives:   {metrics['true_negatives']}")
    print(f"  False Positives:  {metrics['false_positives']}")
    print(f"  False Negatives:  {metrics['false_negatives']}")
    
    print(f"\nError Rates:")
    print(f"  FPR: {metrics['false_positive_rate']:.4f}")
    print(f"  FNR: {metrics['false_negative_rate']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    
    print(f"{'='*60}\n")


def compute_financial_impact(
    metrics: Dict,
    avg_transaction_value: float = 100.0,
    fraud_cost_multiplier: float = 3.0
) -> Dict:
    """
    Compute financial impact of fraud detection
    
    Args:
        metrics: Detection metrics
        avg_transaction_value: Average transaction value
        fraud_cost_multiplier: Cost multiplier for fraud (investigation, chargebacks, etc.)
    
    Returns:
        Dictionary of financial metrics
    """
    tp = metrics['true_positives']
    fp = metrics['false_positives']
    fn = metrics['false_negatives']
    
    # Prevented fraud losses
    prevented_fraud = tp * avg_transaction_value
    
    # Missed fraud losses
    missed_fraud = fn * avg_transaction_value
    
    # False alarm costs (investigation)
    false_alarm_cost = fp * avg_transaction_value * 0.1  # 10% of transaction value
    
    # Total cost of fraud (with multiplier for secondary costs)
    total_fraud_cost = missed_fraud * fraud_cost_multiplier
    
    # Net benefit
    net_benefit = prevented_fraud - false_alarm_cost - total_fraud_cost
    
    financial_metrics = {
        'prevented_fraud_amount': prevented_fraud,
        'missed_fraud_amount': missed_fraud,
        'false_alarm_cost': false_alarm_cost,
        'total_fraud_cost': total_fraud_cost,
        'net_benefit': net_benefit,
        'roi': (prevented_fraud / (false_alarm_cost + 1e-6)) if false_alarm_cost > 0 else float('inf')
    }
    
    return financial_metrics
