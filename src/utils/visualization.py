"""
Visualization utilities for federated learning
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_history(history: List[Dict], save_path: str = None):
    """
    Plot training history metrics
    
    Args:
        history: List of training round statistics
        save_path: Path to save the plot
    """
    if not history:
        logger.warning("No history to plot")
        return
    
    rounds = [h['round'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss over rounds
    if 'train_loss' in history[0]:
        losses = [h.get('train_loss', 0) for h in history]
        axes[0, 0].plot(rounds, losses, marker='o', linewidth=2)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss over Rounds')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over rounds
    if 'accuracy' in history[0]:
        accuracies = [h.get('accuracy', 0) for h in history]
        axes[0, 1].plot(rounds, accuracies, marker='s', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy over Rounds')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Privacy budget
    if 'epsilon_spent' in history[0]:
        epsilons = [h.get('epsilon_spent', 0) for h in history]
        axes[1, 0].plot(rounds, epsilons, marker='^', color='red', linewidth=2)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Budget Limit')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Epsilon Spent')
        axes[1, 0].set_title('Privacy Budget over Rounds')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Number of clients
    if 'num_clients' in history[0]:
        num_clients = [h.get('num_clients', 0) for h in history]
        axes[1, 1].plot(rounds, num_clients, marker='d', color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Number of Clients')
        axes[1, 1].set_title('Participating Clients over Rounds')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_client_data_distribution(client_data: List[tuple], save_path: str = None):
    """
    Plot data distribution across clients
    
    Args:
        client_data: List of (X, y) tuples for each client
        save_path: Path to save the plot
    """
    num_clients = len(client_data)
    
    client_sizes = [len(y) for _, y in client_data]
    fraud_ratios = [y.mean() for _, y in client_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Dataset sizes
    axes[0].bar(range(num_clients), client_sizes, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Client ID')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Dataset Size per Client')
    axes[0].set_xticks(range(num_clients))
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Fraud ratios
    axes[1].bar(range(num_clients), fraud_ratios, color='coral', alpha=0.7)
    axes[1].set_xlabel('Client ID')
    axes[1].set_ylabel('Fraud Ratio')
    axes[1].set_title('Fraud Ratio per Client')
    axes[1].set_xticks(range(num_clients))
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Client distribution plot saved to {save_path}")
    
    plt.show()


def plot_privacy_accuracy_tradeoff(
    epsilons: List[float],
    accuracies: List[float],
    save_path: str = None
):
    """
    Plot privacy-accuracy tradeoff
    
    Args:
        epsilons: List of epsilon values
        accuracies: Corresponding accuracies
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Privacy-Accuracy Tradeoff', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Highlight the sweet spot
    if accuracies:
        best_idx = np.argmax(accuracies)
        plt.scatter([epsilons[best_idx]], [accuracies[best_idx]], 
                   color='red', s=200, marker='*', zorder=5,
                   label=f'Best: ε={epsilons[best_idx]:.2f}, Acc={accuracies[best_idx]:.4f}')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Privacy-accuracy tradeoff plot saved to {save_path}")
    
    plt.show()
