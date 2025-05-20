"""
Unit tests for data preprocessing module.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

# Add the source directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock data
        self.user_data_path = 'tests/mock_user_data.csv'
        self.message_data_path = 'tests/mock_message_data.csv'
        
        # Create mock user data
        user_data = pd.DataFrame({
            'uid': [0, 1, 2, 3, 4],
            'user_profile': [
                'I love traveling to different countries.',
                'Exploring new cultures is my passion.',
                'I enjoy trying local cuisine when I travel.',
                'The beaches in Hawaii are breathtaking.',
                'I had a great time visiting museums in Paris.'
            ]
        })
        
        # Create mock message data
        message_data = pd.DataFrame({
            'message': [
                'Let\'s explore the world together!',
                'Join me for an adventure in Italy.',
                'Would you like to visit the Grand Canyon?',
                'Come with me to Japan next summer.',
                'Let\'s go hiking in the mountains.'
            ],
            'sid': [0, 1, 2, 3, 4],
            'rid': [1, 2, 3, 4, 0],
            'success': [1, 0, 1, 1, 0]
        })
        
        # Save mock data
        os.makedirs('tests', exist_ok=True)
        user_data.to_csv(self.user_data_path, index=False)
        message_data.to_csv(self.message_data_path, index=False)
        
        # Create preprocessor
        self.preprocessor = DataPreprocessor(self.user_data_path, self.message_data_path)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove mock data files
        if os.path.exists(self.user_data_path):
            os.remove(self.user_data_path)
        if os.path.exists(self.message_data_path):
            os.remove(self.message_data_path)
        
        # Remove tests directory if empty
        try:
            os.rmdir('tests')
        except OSError:
            pass
    
    def test_load_data(self):
        """Test loading data from CSV files."""
        user_data, message_data = self.preprocessor.load_data()
        
        # Check that data was loaded correctly
        self.assertEqual(len(user_data), 5)
        self.assertEqual(len(message_data), 5)
        self.assertEqual(user_data.shape[1], 2)
        self.assertEqual(message_data.shape[1], 4)
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "I love traveling to different countries! It's amazing."
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Check that text was cleaned correctly
        self.assertNotIn('!', cleaned_text)
        self.assertNotIn('\'', cleaned_text)
        self.assertEqual(cleaned_text, 'love traveling different countries amazing')
    
    def test_preprocess_user_data(self):
        """Test preprocessing user data."""
        self.preprocessor.load_data()
        user_data = self.preprocessor.preprocess_user_data()
        
        # Check that user data was preprocessed correctly
        self.assertIn('cleaned_profile', user_data.columns)
        self.assertEqual(len(user_data), 5)
        self.assertNotEqual(user_data['user_profile'][0], user_data['cleaned_profile'][0])
    
    def test_preprocess_message_data(self):
        """Test preprocessing message data."""
        self.preprocessor.load_data()
        message_data = self.preprocessor.preprocess_message_data()
        
        # Check that message data was preprocessed correctly
        self.assertIn('cleaned_message', message_data.columns)
        self.assertEqual(len(message_data), 5)
        self.assertNotEqual(message_data['message'][0], message_data['cleaned_message'][0])
    
    def test_create_graph(self):
        """Test creating graph from user and message data."""
        self.preprocessor.load_data()
        graph = self.preprocessor.create_graph()
        
        # Check that graph was created correctly
        self.assertEqual(len(graph.nodes), 5)
        self.assertEqual(len(graph.edges), 5)
        
        # Check that edges have correct attributes
        for u, v, data in graph.edges(data=True):
            self.assertIn('success', data)
            self.assertIn(data['success'], [0, 1])
    
    def test_convert_to_pytorch_geometric(self):
        """Test converting graph to PyTorch Geometric format."""
        self.preprocessor.load_data()
        graph = self.preprocessor.create_graph()
        
        # Create mock node features
        node_features = {node: np.random.rand(4) for node in graph.nodes}
        
        # Convert to PyTorch Geometric format
        data = self.preprocessor.convert_to_pytorch_geometric(node_features)
        
        # Check that data was converted correctly
        self.assertIsInstance(data, Data)
        self.assertEqual(data.x.shape, (5, 4))
        self.assertEqual(data.edge_index.shape, (2, 5))
        self.assertEqual(data.edge_attr.shape, (5, 1))

if __name__ == '__main__':
    unittest.main()
