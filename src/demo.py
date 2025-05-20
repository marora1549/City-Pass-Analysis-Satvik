"""
Demonstration script for city pass GNN model.

This script loads a pre-trained model and demonstrates prediction
of invitation success on sample data.
"""
import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add the source directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.sentiment_analysis import SentimentAnalyzer, FeatureExtractor
from src.gnn_model import GCNModel, GATModel, GraphSAGEModel
from src.utils import plot_graph

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Demo of city pass GNN model')
    
    parser.add_argument('--user_data', type=str, default='data/user_data.csv',
                        help='Path to user data CSV file')
    parser.add_argument('--message_data', type=str, default='data/message_data.csv',
                        help='Path to message data CSV file')
    parser.add_argument('--model_path', type=str, default='output/gcn_model.pt',
                        help='Path to saved model')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sage'],
                        help='Type of GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Dimension of hidden layers')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    return parser.parse_args()

def main():
    """
    Main function to demonstrate the model.
    """
    # Parse arguments
    args = parse_args()
    
    # Set device
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
    sentiment_analyzer = SentimentAnalyzer(method='textblob')
    feature_extractor = FeatureExtractor(sentiment_analyzer)
    
    # Extract features from user profiles
    user_data['sentiment_score'] = sentiment_analyzer.analyze_batch(user_data['cleaned_profile'])
    user_features = feature_extractor.extract_batch_features(user_data['cleaned_profile'])
    
    # Extract features from messages
    message_data['sentiment_score'] = sentiment_analyzer.analyze_batch(message_data['cleaned_message'])
    
    # Step 3: Create graph
    print('Step 3: Creating graph...')
    graph = preprocessor.create_graph()
    
    # Create node features dictionary
    node_features = {user_id: user_features[i] for i, user_id in enumerate(user_data['uid'])}
    
    # Convert to PyTorch Geometric data
    data = preprocessor.convert_to_pytorch_geometric(node_features)
    
    # Step 4: Load model
    print('Step 4: Loading model...')
    
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
    
    # Load model state
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Loaded model from {args.model_path}')
    else:
        print(f'Warning: Model file {args.model_path} not found. Using untrained model.')
    
    model = model.to(device)
    model.eval()
    
    # Step 5: Make predictions
    print('Step 5: Making predictions...')
    
    # Get data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    # Get node embeddings
    with torch.no_grad():
        node_embeddings = model.encode(x, edge_index)
    
    # Step 6: Visualize results
    print('Step 6: Visualizing results...')
    
    # Create NetworkX graph for visualization
    G = to_networkx(data, to_undirected=True)
    
    # Sample subgraph for clearer visualization
    if len(G.nodes()) > 50:
        print(f'Sampling 50 nodes out of {len(G.nodes())} for visualization...')
        sampled_nodes = list(G.nodes())[:50]
        G = G.subgraph(sampled_nodes)
    
    # Get predictions for edges in the graph
    edge_list = list(G.edges())
    edge_index_pred = torch.tensor([[src, dst] for src, dst in edge_list], dtype=torch.long).t().contiguous().to(device)
    
    with torch.no_grad():
        pred = model.decode(node_embeddings, edge_index_pred)
    
    # Get prediction values
    pred_values = pred.cpu().numpy().flatten()
    
    # Color edges based on prediction values
    edge_colors = ['green' if p > 0.5 else 'red' for p in pred_values]
    
    # Visualize graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.8, node_color='skyblue')
    
    # Draw edges with prediction colors
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, arrows=True)
    
    plt.title(f'{args.model_type.upper()} Model Predictions (Green: Success, Red: Failure)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'prediction_visualization.png'))
    
    print(f'Visualization saved to {os.path.join(args.output_dir, "prediction_visualization.png")}')
    
    # Step 7: Sample predictions
    print('\nStep 7: Sample predictions:')
    
    # Select a few sample edges
    sample_indices = np.random.choice(len(edge_list), min(5, len(edge_list)), replace=False)
    
    print('\nSample predictions:')
    print('-' * 60)
    print(f'{"From User":<10} {"To User":<10} {"Prediction":<10} {"Probability":<15}')
    print('-' * 60)
    
    for i in sample_indices:
        src, dst = edge_list[i]
        prob = pred_values[i]
        prediction = 'Success' if prob > 0.5 else 'Failure'
        
        print(f'{src:<10} {dst:<10} {prediction:<10} {prob:.4f}')
    
    print('-' * 60)
    print('Done!')

if __name__ == '__main__':
    main()
