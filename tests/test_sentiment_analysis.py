"""
Unit tests for sentiment analysis module.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from textblob import TextBlob

# Add the source directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analysis import SentimentAnalyzer, FeatureExtractor

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sentiment analyzers for different methods
        self.textblob_analyzer = SentimentAnalyzer(method='textblob')
        self.vader_analyzer = SentimentAnalyzer(method='vader')
        
        # Sample texts with different sentiments
        self.positive_text = "I love traveling and experiencing new cultures. It's amazing!"
        self.negative_text = "I hated that trip. The weather was terrible and the hotel was dirty."
        self.neutral_text = "We went to Paris last summer. It was a vacation."
    
    def test_analyze_sentiment_textblob(self):
        """Test sentiment analysis using TextBlob."""
        # Test positive text
        positive_score = self.textblob_analyzer.analyze_sentiment_textblob(self.positive_text)
        self.assertGreater(positive_score, 0)
        
        # Test negative text
        negative_score = self.textblob_analyzer.analyze_sentiment_textblob(self.negative_text)
        self.assertLess(negative_score, 0)
        
        # Test neutral text
        neutral_score = self.textblob_analyzer.analyze_sentiment_textblob(self.neutral_text)
        self.assertAlmostEqual(abs(neutral_score), 0, delta=0.3)
    
    def test_analyze_sentiment_vader(self):
        """Test sentiment analysis using VADER."""
        # Test positive text
        positive_score = self.vader_analyzer.analyze_sentiment_vader(self.positive_text)
        self.assertGreater(positive_score, 0)
        
        # Test negative text
        negative_score = self.vader_analyzer.analyze_sentiment_vader(self.negative_text)
        self.assertLess(negative_score, 0)
    
    def test_analyze_sentiment(self):
        """Test the main analyze_sentiment method."""
        # Test with TextBlob
        score_textblob = self.textblob_analyzer.analyze_sentiment(self.positive_text)
        self.assertGreater(score_textblob, 0)
        
        # Test with VADER
        score_vader = self.vader_analyzer.analyze_sentiment(self.positive_text)
        self.assertGreater(score_vader, 0)
        
        # Test with empty text
        score_empty = self.textblob_analyzer.analyze_sentiment("")
        self.assertEqual(score_empty, 0.0)
        
        # Test with None
        score_none = self.textblob_analyzer.analyze_sentiment(None)
        self.assertEqual(score_none, 0.0)
    
    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        texts = [self.positive_text, self.negative_text, self.neutral_text]
        scores = self.textblob_analyzer.analyze_batch(texts)
        
        # Check that we got the right number of scores
        self.assertEqual(len(scores), 3)
        
        # Check that scores are in the expected order
        self.assertGreater(scores[0], scores[1])


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sentiment analyzer and feature extractor
        self.sentiment_analyzer = SentimentAnalyzer(method='textblob')
        self.feature_extractor = FeatureExtractor(self.sentiment_analyzer)
        
        # Sample text
        self.text = "I love traveling and experiencing new cultures. It's amazing!"
    
    def test_extract_features(self):
        """Test feature extraction from text."""
        features = self.feature_extractor.extract_features(self.text)
        
        # Check that features have the expected shape
        self.assertEqual(len(features), 4)
        
        # Check that sentiment score is positive
        self.assertGreater(features[0], 0)
        
        # Check that subjectivity is between 0 and 1
        self.assertGreaterEqual(features[1], 0)
        self.assertLessEqual(features[1], 1)
        
        # Check that word count and character count are positive
        self.assertGreater(features[2], 0)
        self.assertGreater(features[3], 0)
    
    def test_extract_batch_features(self):
        """Test batch feature extraction."""
        texts = [self.text, "I hate traveling.", "Paris is a city in France."]
        features = self.feature_extractor.extract_batch_features(texts)
        
        # Check that features have the expected shape
        self.assertEqual(features.shape, (3, 4))

if __name__ == '__main__':
    unittest.main()
