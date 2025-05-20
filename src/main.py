"""
Main script for city pass sentiment analysis and invitation success prediction.

This script brings together all components of the project to perform
sentiment analysis on user profiles and messages, and to train a
Graph Neural Network to predict invitation success.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Add the source directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.sentiment_analysis import SentimentAnalyzer, FeatureExtractor
from src.gnn_model import GCNModel, GATModel, GraphSAGEModel, GNNTrainer
from src.utils import (
    plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve,
    plot_graph, plot_sentiment_distribution, plot_feature_importance,
    plot_success_rate_by_sentiment
)

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='City pass sentiment analysis and invitation success prediction')
    
    # Data paths
    parser.add_argument('--user_data', type=str, default='data/user_data.csv',
                        help='Path to user data CSV file')
    parser.add_argument('--message_data', type=str, default='data/message_data.csv',
                        help='Path to message data CSV file')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sage'],
                        help='Type of GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Dimension of hidden layers')
    parser.add_argument('--sentiment_method', type=str, default='textblob',
                        choices=['textblob', 'vader', 'transformer', 'ml'],
                        help='Method to use for sentiment analysis')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of epochs to wait for improvement')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()

def main():
    """
    Main function to run the pipeline.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device('cpu') if args.no_cuda or not torch.cuda.is_available() else torch.device('cuda')
    print(f'Using device: {device}')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print('Step 1: Loading and preprocessing data...')
    preprocessor = DataPreprocessor(args.user_data, args.message_data)
    preprocessor.load_data()
    
    user_data = preprocessor.preprocess_user_data()
    message_data = preprocessor.preprocess_message_data()
    
    # Step 2: Perform sentiment analysis
    print('Step 2: Performing sentiment analysis...')
    sentiment_analyzer = SentimentAnalyzer(method=args.sentiment_method)
    feature_extractor = FeatureExtractor(sentiment_analyzer)
    
    # Extract features from user profiles
    user_data['sentiment_score'] = sentiment_analyzer.analyze_batch(user_data['cleaned_profile'])
    user_features = feature_extractor.extract_batch_features(user_data['cleaned_profile'])
    
    # Extract features from messages
    message_data['sentiment_score'] = sentiment_analyzer.analyze_batch(message_data['cleaned_message'])
    
    # Step 3: Create graph and prepare data for GNN
    print('Step 3: Creating graph and preparing data for GNN...')
    graph = preprocessor.create_graph()
    
    # Create node features dictionary
    node_features = {user_id: user_features[i] for i, user_id in enumerate(user_data['uid'])}
    
    # Convert to PyTorch Geometric data
    data = preprocessor.convert_to_pytorch_geometric(node_features)
    
    # Step 4: Split data for training, validation, and testing
    print('Step 4: Splitting data...')
    
    # Get edge indices and attributes
    edge_index = data.edge_index.t().tolist()
    edge_attr = data.edge_attr.tolist()
    
    # Split edges
    train_idx, test_idx = train_test_split(
        range(len(edge_index)), test_size=0.2, random_state=args.seed
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, random_state=args.seed
    )
    
    # Create data objects for train, validation, and test
    train_data = Data(
        x=data.x.clone(),
        edge_index=torch.tensor([
            [edge_index[i][0], edge_index[i][1]] for i in train_idx
        ], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor([
            edge_attr[i] for i in train_idx
        ], dtype=torch.float)
    )
    
    val_data = Data(
        x=data.x.clone(),
        edge_index=torch.tensor([
            [edge_index[i][0], edge_index[i][1]] for i in val_idx
        ], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor([
            edge_attr[i] for i in val_idx
        ], dtype=torch.float)
    )
    
    test_data = Data(
        x=data.x.clone(),
        edge_index=torch.tensor([
            [edge_index[i][0], edge_index[i][1]] for i in test_idx
        ], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor([
            edge_attr[i] for i in test_idx
        ], dtype=torch.float)
    )
    
    # Step 5: Create and train GNN model
    print('Step 5: Creating and training GNN model...')
    
    # Get input dimension
    input_dim = data.x.shape[1]
    
    # Create model based on type
    if args.model_type == 'gcn':
        model = GCNModel(input_dim=input_dim, hidden_dim=args.hidden_dim)
    elif args.model_type == 'gat':
        model = GATModel(input_dim=input_dim, hidden_dim=args.hidden_dim)
    elif args.model_type == 'sage':
        model = GraphSAGEModel(input_dim=input_dim, hidden_dim=args.hidden_dim)
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
    
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    # Create trainer
    trainer = GNNTrainer(model, optimizer, criterion, device=device)
    
    # Train model
    metrics = trainer.train(
        train_data, val_data, num_epochs=args.epochs,
        early_stopping=args.early_stopping
    )
    
    # Step 6: Evaluate model
    print('Step 6: Evaluating model...')
    eval_metrics = trainer.evaluate(test_data)
    
    # Print evaluation metrics
    print('Evaluation metrics:')
    for metric, value in eval_metrics.items():
        print(f'  {metric}: {value}')
    
    # Step 7: Generate visualizations
    print('Step 7: Generating visualizations...')
    
    # Plot training metrics
    trainer.plot_training_metrics(save_path=os.path.join(args.output_dir, 'training_metrics.png'))
    
    # Make predictions on test data
    y_pred = trainer.predict(test_data)
    y_true = test_data.edge_attr.cpu().numpy()
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, (y_pred > 0.5).astype(float),
        labels=['Failure', 'Success'],
        title=f'{args.model_type.upper()} Model Confusion Matrix',
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        y_true, y_pred,
        title=f'{args.model_type.upper()} Model ROC Curve',
        save_path=os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # Plot precision-recall curve
    plot_precision_recall_curve(
        y_true, y_pred,
        title=f'{args.model_type.upper()} Model Precision-Recall Curve',
        save_path=os.path.join(args.output_dir, 'precision_recall_curve.png')
    )
    
    # Plot sentiment distribution
    plot_sentiment_distribution(
        user_data['sentiment_score'],
        title='User Profile Sentiment Distribution',
        save_path=os.path.join(args.output_dir, 'user_sentiment_distribution.png')
    )
    
    plot_sentiment_distribution(
        message_data['sentiment_score'],
        title='Message Sentiment Distribution',
        save_path=os.path.join(args.output_dir, 'message_sentiment_distribution.png')
    )
    
    # Plot success rate by sentiment
    plot_success_rate_by_sentiment(
        message_data['sentiment_score'], message_data['success'],
        title='Invitation Success Rate by Message Sentiment',
        save_path=os.path.join(args.output_dir, 'success_by_message_sentiment.png')
    )
    
    # Step 8: Save results
    print('Step 8: Saving results...')
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model_type}_model.pt'))
    
    # Save evaluation metrics
    pd.DataFrame([eval_metrics]).to_csv(os.path.join(args.output_dir, 'evaluation_metrics.csv'), index=False)
    
    # Save training metrics
    pd.DataFrame({
        'epoch': list(range(1, len(metrics['train_losses']) + 1)),
        'train_loss': metrics['train_losses'],
        'val_loss': metrics['val_losses'],
        'train_acc': metrics['train_accs'],
        'val_acc': metrics['val_accs']
    }).to_csv(os.path.join(args.output_dir, 'training_metrics.csv'), index=False)
    
    print('Done!')

if __name__ == '__main__':
    main()
