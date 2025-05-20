"""
Utility module for evaluation and visualization.

This module provides functions for evaluating models
and visualizing results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc
)
import networkx as nx
import torch
from torch_geometric.utils import to_networkx

def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', cmap='Blues', save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): List of class labels. Default: None
        title (str): Plot title. Default: 'Confusion Matrix'
        cmap (str): Color map for the plot. Default: 'Blues'
        save_path (str): Path to save the plot. Default: None
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_precision_recall_curve(y_true, y_score, title='Precision-Recall Curve', save_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true (array-like): True labels.
        y_score (array-like): Predicted scores.
        title (str): Plot title. Default: 'Precision-Recall Curve'
        save_path (str): Path to save the plot. Default: None
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'AUC = {auc(recall, precision):.3f}')
    
    # Set labels
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_roc_curve(y_true, y_score, title='ROC Curve', save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true (array-like): True labels.
        y_score (array-like): Predicted scores.
        title (str): Plot title. Default: 'ROC Curve'
        save_path (str): Path to save the plot. Default: None
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--')
    
    # Set labels
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Add grid
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_graph(data, node_color=None, edge_color=None, node_size=100, title='Graph Visualization', save_path=None):
    """
    Plot a graph from PyTorch Geometric data.
    
    Args:
        data (torch_geometric.data.Data): PyTorch Geometric data.
        node_color (array-like): Node colors. Default: None
        edge_color (array-like): Edge colors. Default: None
        node_size (int): Size of nodes. Default: 100
        title (str): Plot title. Default: 'Graph Visualization'
        save_path (str): Path to save the plot. Default: None
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set default colors if not provided
    if node_color is None:
        node_color = 'skyblue'
    
    if edge_color is None:
        edge_color = 'gray'
    
    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=0.5, ax=ax)
    
    # Add title
    ax.set_title(title)
    
    # Turn off axis
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_sentiment_distribution(sentiment_scores, title='Sentiment Distribution', save_path=None):
    """
    Plot distribution of sentiment scores.
    
    Args:
        sentiment_scores (array-like): Sentiment scores.
        title (str): Plot title. Default: 'Sentiment Distribution'
        save_path (str): Path to save the plot. Default: None
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(sentiment_scores, kde=True, ax=ax)
    
    # Set labels
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_feature_importance(feature_names, feature_importances, title='Feature Importance', save_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_names (list): List of feature names.
        feature_importances (array-like): Feature importance values.
        title (str): Plot title. Default: 'Feature Importance'
        save_path (str): Path to save the plot. Default: None
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    sns.barplot(x=np.array(feature_importances)[indices], y=np.array(feature_names)[indices], ax=ax)
    
    # Set labels
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def plot_success_rate_by_sentiment(sentiment_scores, success_labels, bins=10, title='Success Rate by Sentiment', save_path=None):
    """
    Plot success rate by sentiment score.
    
    Args:
        sentiment_scores (array-like): Sentiment scores.
        success_labels (array-like): Success labels (0 or 1).
        bins (int): Number of bins for sentiment scores. Default: 10
        title (str): Plot title. Default: 'Success Rate by Sentiment'
        save_path (str): Path to save the plot. Default: None
        
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    # Create DataFrame
    df = pd.DataFrame({
        'sentiment': sentiment_scores,
        'success': success_labels
    })
    
    # Create bins
    df['sentiment_bin'] = pd.cut(df['sentiment'], bins=bins)
    
    # Calculate success rate by bin
    success_rate = df.groupby('sentiment_bin')['success'].mean().reset_index()
    count = df.groupby('sentiment_bin').size().reset_index(name='count')
    result = pd.merge(success_rate, count, on='sentiment_bin')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bar chart
    bars = ax.bar(range(len(result)), result['success'], alpha=0.7)
    
    # Set labels
    ax.set_xlabel('Sentiment Score')
    ax.set_ylabel('Success Rate')
    ax.set_title(title)
    ax.set_xticks(range(len(result)))
    ax.set_xticklabels([str(bin_range) for bin_range in result['sentiment_bin']], rotation=45)
    
    # Add count labels
    for i, (bar, count_val) in enumerate(zip(bars, result['count'])):
        ax.text(i, bar.get_height() + 0.02, f'n={count_val}', 
                ha='center', va='bottom', fontsize=8)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig
