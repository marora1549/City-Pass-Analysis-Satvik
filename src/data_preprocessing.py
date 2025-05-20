"""
Data preprocessing module for city pass sentiment analysis.

This module handles loading, cleaning, and preprocessing of user data
and message data for sentiment analysis and graph construction.
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import networkx as nx
import torch
from torch_geometric.data import Data

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class DataPreprocessor:
    """
    Class for preprocessing user and message data.
    
    This class handles loading data from CSV files, cleaning text,
    extracting features, and constructing a graph for GNN analysis.
    """
    
    def __init__(self, user_data_path, message_data_path):
        """
        Initialize the DataPreprocessor.
        
        Args:
            user_data_path (str): Path to the user data CSV file.
            message_data_path (str): Path to the message data CSV file.
        """
        self.user_data_path = user_data_path
        self.message_data_path = message_data_path
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Will be populated when load_data is called
        self.user_data = None
        self.message_data = None
        self.graph = None
        
    def load_data(self):
        """
        Load user and message data from CSV files.
        
        Returns:
            tuple: Tuple containing user_data and message_data DataFrames.
        """
        self.user_data = pd.read_csv(self.user_data_path)
        self.message_data = pd.read_csv(self.message_data_path)
        
        # Basic cleaning
        self.user_data = self.user_data.fillna("")
        self.message_data = self.message_data.fillna("")
        
        return self.user_data, self.message_data
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, stopwords, and lemmatizing.
        
        Args:
            text (str): The text to clean.
            
        Returns:
            str: Cleaned text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Join tokens back into a string
        cleaned_text = ' '.join(cleaned_tokens)
        
        return cleaned_text
    
    def preprocess_user_data(self):
        """
        Preprocess user data by cleaning text and extracting features.
        
        Returns:
            pandas.DataFrame: Preprocessed user data.
        """
        if self.user_data is None:
            self.load_data()
        
        # Clean user profiles
        self.user_data['cleaned_profile'] = self.user_data['user_profile'].apply(self.clean_text)
        
        return self.user_data
    
    def preprocess_message_data(self):
        """
        Preprocess message data by cleaning text and extracting features.
        
        Returns:
            pandas.DataFrame: Preprocessed message data.
        """
        if self.message_data is None:
            self.load_data()
        
        # Clean messages
        self.message_data['cleaned_message'] = self.message_data['message'].apply(self.clean_text)
        
        return self.message_data
    
    def create_graph(self):
        """
        Create a graph from user and message data.
        
        Returns:
            networkx.Graph: Graph representing user connections.
        """
        if self.user_data is None or self.message_data is None:
            self.load_data()
            self.preprocess_user_data()
            self.preprocess_message_data()
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes (users)
        for user_id in self.user_data['uid']:
            G.add_node(user_id)
        
        # Add edges (messages)
        for _, row in self.message_data.iterrows():
            sender_id = row['sid']
            receiver_id = row['rid']
            success = row['success']
            
            # Only add edges if both users exist in the graph
            if sender_id in G.nodes and receiver_id in G.nodes:
                G.add_edge(sender_id, receiver_id, success=success)
        
        self.graph = G
        return G
    
    def convert_to_pytorch_geometric(self, node_features):
        """
        Convert networkx graph to PyTorch Geometric format.
        
        Args:
            node_features (dict): Dictionary mapping node IDs to feature vectors.
            
        Returns:
            torch_geometric.data.Data: PyTorch Geometric data object.
        """
        if self.graph is None:
            self.create_graph()
        
        # Convert node IDs to indices
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        
        # Prepare edge index
        edges = list(self.graph.edges())
        edge_index = torch.tensor([[node_mapping[src], node_mapping[dst]] for src, dst in edges], dtype=torch.long).t().contiguous()
        
        # Prepare node features
        x = torch.tensor([node_features[node] for node in self.graph.nodes()], dtype=torch.float)
        
        # Prepare edge labels (success/failure)
        edge_attr = torch.tensor([self.graph.edges[src, dst]['success'] for src, dst in edges], dtype=torch.float).view(-1, 1)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data
