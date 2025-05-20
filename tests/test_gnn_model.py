"""
Unit tests for GNN models.
"""
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data

# Add the source directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gnn_model import GCNModel, GATModel, GraphSAGEModel, GNNTrainer

class TestGNNModels(unittest.TestCase):
    """Test cases for GNN models."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create mock data
        self.num_nodes = 10
        self.input_dim = 4
        self.hidden_dim = 8
        
        # Create random node features
        self.x = torch.randn(self.num_nodes, self.input_dim)
        
        # Create random edges
        edge_index = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and np.random.rand() < 0.3:
                    edge_index.append([i, j])
        
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create random edge attributes (binary)
        self.edge_attr = torch.tensor(
            [[np.random.randint(0, 2)] for _ in range(self.edge_index.shape[1])],
            dtype=torch.float
        )
        
        # Create PyTorch Geometric data object
        self.data = Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr
        )
        
        # Create models
        self.gcn_model = GCNModel(self.input_dim, self.hidden_dim)
        self.gat_model = GATModel(self.input_dim, self.hidden_dim)
        self.sage_model = GraphSAGEModel(self.input_dim, self.hidden_dim)
    
    def test_gcn_model(self):
        """Test GCN model."""
        # Test forward pass
        out = self.gcn_model(self.x, self.edge_index)
        
        # Check output shape
        self.assertEqual(out.shape, (self.edge_index.shape[1], 1))
        
        # Check output values are probabilities (between 0 and 1)
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())
        
        # Test node embedding
        z = self.gcn_model.encode(self.x, self.edge_index)
        self.assertEqual(z.shape, (self.num_nodes, self.hidden_dim))
    
    def test_gat_model(self):
        """Test GAT model."""
        # Test forward pass
        out = self.gat_model(self.x, self.edge_index)
        
        # Check output shape
        self.assertEqual(out.shape, (self.edge_index.shape[1], 1))
        
        # Check output values are probabilities (between 0 and 1)
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())
        
        # Test node embedding
        z = self.gat_model.encode(self.x, self.edge_index)
        self.assertEqual(z.shape, (self.num_nodes, self.hidden_dim))
    
    def test_sage_model(self):
        """Test GraphSAGE model."""
        # Test forward pass
        out = self.sage_model(self.x, self.edge_index)
        
        # Check output shape
        self.assertEqual(out.shape, (self.edge_index.shape[1], 1))
        
        # Check output values are probabilities (between 0 and 1)
        self.assertTrue((out >= 0).all())
        self.assertTrue((out <= 1).all())
        
        # Test node embedding
        z = self.sage_model.encode(self.x, self.edge_index)
        self.assertEqual(z.shape, (self.num_nodes, self.hidden_dim))


class TestGNNTrainer(unittest.TestCase):
    """Test cases for GNNTrainer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create mock data
        self.num_nodes = 10
        self.input_dim = 4
        self.hidden_dim = 8
        
        # Create random node features
        self.x = torch.randn(self.num_nodes, self.input_dim)
        
        # Create random edges
        edge_index = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and np.random.rand() < 0.3:
                    edge_index.append([i, j])
        
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create random edge attributes (binary)
        self.edge_attr = torch.tensor(
            [[np.random.randint(0, 2)] for _ in range(self.edge_index.shape[1])],
            dtype=torch.float
        )
        
        # Create PyTorch Geometric data object
        self.data = Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr
        )
        
        # Create model, optimizer, and criterion
        self.model = GCNModel(self.input_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        
        # Create trainer
        self.trainer = GNNTrainer(self.model, self.optimizer, self.criterion)
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        # Train for one epoch
        loss, acc = self.trainer.train_epoch(self.data)
        
        # Check that loss and accuracy are reasonable
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
    
    def test_validate(self):
        """Test validation."""
        # Validate
        loss, acc = self.trainer.validate(self.data)
        
        # Check that loss and accuracy are reasonable
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)
    
    def test_train(self):
        """Test training for multiple epochs."""
        # Train for 2 epochs
        metrics = self.trainer.train(self.data, self.data, num_epochs=2, early_stopping=5)
        
        # Check that metrics were tracked correctly
        self.assertEqual(len(metrics['train_losses']), 2)
        self.assertEqual(len(metrics['val_losses']), 2)
        self.assertEqual(len(metrics['train_accs']), 2)
        self.assertEqual(len(metrics['val_accs']), 2)
    
    def test_evaluate(self):
        """Test evaluation."""
        # Evaluate
        metrics = self.trainer.evaluate(self.data)
        
        # Check that metrics were calculated correctly
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
    
    def test_predict(self):
        """Test prediction."""
        # Make predictions
        preds = self.trainer.predict(self.data)
        
        # Check that predictions have the expected shape
        self.assertEqual(preds.shape, (self.edge_index.shape[1], 1))
        
        # Check that predictions are probabilities (between 0 and 1)
        self.assertTrue((preds >= 0).all())
        self.assertTrue((preds <= 1).all())

if __name__ == '__main__':
    unittest.main()
