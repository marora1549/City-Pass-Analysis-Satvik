"""
Graph Neural Network module for city pass analysis.

This module implements various GNN architectures for predicting
invitation success between users.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

class GCNModel(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model.
    
    This model uses GCN layers to learn node representations
    and predict edge properties.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        """
        Initialize the GCN model.
        
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers. Default: 64
            output_dim (int): Dimension of output. Default: 1
        """
        super(GCNModel, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        
    def encode(self, x, edge_index):
        """
        Encode nodes using GCN layers.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """
        Decode node embeddings to predict edge properties.
        
        Args:
            z (torch.Tensor): Node embeddings.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Edge predictions.
        """
        # Get node features for both source and target nodes
        src, dst = edge_index
        src_z = z[src]
        dst_z = z[dst]
        
        # Concatenate source and target node embeddings
        edge_features = torch.cat([src_z, dst_z], dim=1)
        
        # Apply MLP to get edge predictions
        x = self.lin1(edge_features)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        
        return x
    
    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Edge predictions.
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)


class GATModel(torch.nn.Module):
    """
    Graph Attention Network (GAT) model.
    
    This model uses GAT layers to learn node representations
    and predict edge properties.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, heads=8):
        """
        Initialize the GAT model.
        
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers. Default: 64
            output_dim (int): Dimension of output. Default: 1
            heads (int): Number of attention heads. Default: 8
        """
        super(GATModel, self).__init__()
        
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        
    def encode(self, x, edge_index):
        """
        Encode nodes using GAT layers.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """
        Decode node embeddings to predict edge properties.
        
        Args:
            z (torch.Tensor): Node embeddings.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Edge predictions.
        """
        # Get node features for both source and target nodes
        src, dst = edge_index
        src_z = z[src]
        dst_z = z[dst]
        
        # Concatenate source and target node embeddings
        edge_features = torch.cat([src_z, dst_z], dim=1)
        
        # Apply MLP to get edge predictions
        x = self.lin1(edge_features)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        
        return x
    
    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Edge predictions.
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)


class GraphSAGEModel(torch.nn.Module):
    """
    GraphSAGE model.
    
    This model uses GraphSAGE layers to learn node representations
    and predict edge properties.
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        """
        Initialize the GraphSAGE model.
        
        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers. Default: 64
            output_dim (int): Dimension of output. Default: 1
        """
        super(GraphSAGEModel, self).__init__()
        
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        
    def encode(self, x, edge_index):
        """
        Encode nodes using GraphSAGE layers.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """
        Decode node embeddings to predict edge properties.
        
        Args:
            z (torch.Tensor): Node embeddings.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Edge predictions.
        """
        # Get node features for both source and target nodes
        src, dst = edge_index
        src_z = z[src]
        dst_z = z[dst]
        
        # Concatenate source and target node embeddings
        edge_features = torch.cat([src_z, dst_z], dim=1)
        
        # Apply MLP to get edge predictions
        x = self.lin1(edge_features)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        
        return x
    
    def forward(self, x, edge_index):
        """
        Forward pass of the GraphSAGE model.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            
        Returns:
            torch.Tensor: Edge predictions.
        """
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

class GNNTrainer:
    """
    Trainer class for Graph Neural Network models.
    
    This class handles training, validation, and testing of GNN models.
    """
    
    def __init__(self, model, optimizer, criterion, device=None):
        """
        Initialize the GNNTrainer.
        
        Args:
            model (torch.nn.Module): GNN model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (callable): Loss function.
            device (torch.device): Device to use for training. Default: None
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
        # Use GPU if available and not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Tracking metrics during training
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, data):
        """
        Train the model for one epoch.
        
        Args:
            data (torch_geometric.data.Data): Training data.
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.train()
        
        # Get data
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.edge_attr.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        out = self.model(x, edge_index)
        loss = self.criterion(out, y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        pred = (out > 0.5).float()
        acc = accuracy_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        
        return loss.item(), acc
    
    def validate(self, data):
        """
        Validate the model.
        
        Args:
            data (torch_geometric.data.Data): Validation data.
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.eval()
        
        # Get data
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.edge_attr.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            out = self.model(x, edge_index)
            loss = self.criterion(out, y)
        
        # Calculate accuracy
        pred = (out > 0.5).float()
        acc = accuracy_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        
        return loss.item(), acc
    
    def train(self, train_data, val_data, num_epochs=100, early_stopping=10):
        """
        Train the model.
        
        Args:
            train_data (torch_geometric.data.Data): Training data.
            val_data (torch_geometric.data.Data): Validation data.
            num_epochs (int): Number of epochs to train for. Default: 100
            early_stopping (int): Number of epochs to wait for improvement. Default: 10
            
        Returns:
            dict: Dictionary of training metrics.
        """
        # Variables for early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_data)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_data)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data (torch_geometric.data.Data): Test data.
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        self.model.eval()
        
        # Get data
        x = test_data.x.to(self.device)
        edge_index = test_data.edge_index.to(self.device)
        y = test_data.edge_attr.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            out = self.model(x, edge_index)
            loss = self.criterion(out, y)
        
        # Convert to numpy for sklearn metrics
        y_true = y.cpu().numpy()
        y_pred_proba = out.cpu().numpy()
        y_pred = (y_pred_proba > 0.5).astype(float)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate ROC-AUC if applicable
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except:
            roc_auc = None
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        return metrics
    
    def predict(self, data):
        """
        Make predictions using the trained model.
        
        Args:
            data (torch_geometric.data.Data): Data to predict on.
            
        Returns:
            numpy.ndarray: Predicted probabilities.
        """
        self.model.eval()
        
        # Get data
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            out = self.model(x, edge_index)
        
        # Return predictions
        return out.cpu().numpy()
    
    def plot_training_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Args:
            save_path (str): Path to save the plot. Default: None
            
        Returns:
            matplotlib.figure.Figure: The created figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
