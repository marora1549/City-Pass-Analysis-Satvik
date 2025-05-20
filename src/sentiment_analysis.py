"""
Sentiment analysis module for city pass analysis.

This module implements various sentiment analysis techniques
for analyzing user profiles and messages.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download NLTK resources if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    Class for sentiment analysis of text data.
    
    This class implements various techniques for sentiment analysis,
    including rule-based, traditional ML, and deep learning approaches.
    """
    
    def __init__(self, method='textblob'):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            method (str): The sentiment analysis method to use.
                Options: 'textblob', 'vader', 'ml', 'transformer'.
                Default: 'textblob'
        """
        self.method = method
        
        # Initialize the appropriate analyzer based on the method
        if self.method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        elif self.method == 'transformer':
            # Using distilbert for efficiency
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        elif self.method == 'ml':
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.classifier = LogisticRegression(max_iter=1000)
        
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            float: Sentiment polarity score [-1.0, 1.0].
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def analyze_sentiment_vader(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            float: Compound sentiment score [-1.0, 1.0].
        """
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def analyze_sentiment_transformer(self, text):
        """
        Analyze sentiment using a pre-trained transformer model.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            float: Sentiment score.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get logits and apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Return score between -1 and 1 (negative to positive)
        # For distilbert-sst2: [0] is negative sentiment, [1] is positive sentiment
        score = (probs[0][1].item() - probs[0][0].item())
        return score
    
    def train_ml_sentiment(self, texts, labels):
        """
        Train a traditional ML model for sentiment analysis.
        
        Args:
            texts (list): List of text samples.
            labels (list): List of sentiment labels (0 for negative, 1 for positive).
            
        Returns:
            tuple: (accuracy, classification_report)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Transform text to features
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def analyze_sentiment_ml(self, text):
        """
        Analyze sentiment using the trained ML model.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            float: Sentiment score between -1 and 1.
        """
        # Transform text
        text_vec = self.vectorizer.transform([text])
        
        # Predict probability of positive sentiment
        proba = self.classifier.predict_proba(text_vec)[0]
        
        # Convert to score between -1 and 1
        score = proba[1] * 2 - 1
        
        return score
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using the selected method.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            float: Sentiment score between -1 and 1.
        """
        if not text or pd.isna(text) or text.strip() == "":
            return 0.0
        
        if self.method == 'textblob':
            return self.analyze_sentiment_textblob(text)
        elif self.method == 'vader':
            return self.analyze_sentiment_vader(text)
        elif self.method == 'transformer':
            return self.analyze_sentiment_transformer(text)
        elif self.method == 'ml':
            return self.analyze_sentiment_ml(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (list): List of text samples.
            
        Returns:
            list: List of sentiment scores.
        """
        return [self.analyze_sentiment(text) for text in texts]

class FeatureExtractor:
    """
    Class for extracting features from text for sentiment analysis.
    
    This class extracts various features from text that can be used
    for sentiment analysis and GNN node representation.
    """
    
    def __init__(self, sentiment_analyzer):
        """
        Initialize the FeatureExtractor.
        
        Args:
            sentiment_analyzer (SentimentAnalyzer): Sentiment analyzer to use.
        """
        self.sentiment_analyzer = sentiment_analyzer
        
    def extract_features(self, text):
        """
        Extract features from text.
        
        Args:
            text (str): The text to extract features from.
            
        Returns:
            numpy.ndarray: Feature vector.
        """
        # Get sentiment score
        sentiment_score = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Get text length features
        word_count = len(text.split())
        char_count = len(text)
        
        # Get additional TextBlob metrics
        blob = TextBlob(text)
        subjectivity = blob.sentiment.subjectivity
        
        # Create feature vector
        features = np.array([
            sentiment_score,
            subjectivity,
            word_count / 100,  # Normalize word count
            char_count / 1000,  # Normalize character count
        ])
        
        return features
    
    def extract_batch_features(self, texts):
        """
        Extract features for a batch of texts.
        
        Args:
            texts (list): List of text samples.
            
        Returns:
            numpy.ndarray: Matrix of feature vectors.
        """
        return np.array([self.extract_features(text) for text in texts])
