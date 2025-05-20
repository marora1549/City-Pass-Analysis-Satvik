"""
Simple script to generate analysis results for city pass invitation prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.metrics import confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
DATA_DIR = 'data'
OUTPUT_DIR = 'output'
USER_DATA_PATH = os.path.join(DATA_DIR, 'user_data.csv')
MESSAGE_DATA_PATH = os.path.join(DATA_DIR, 'message_data.csv')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data...")
user_data = pd.read_csv(USER_DATA_PATH)
message_data = pd.read_csv(MESSAGE_DATA_PATH)

print(f"User data shape: {user_data.shape}")
print(f"Message data shape: {message_data.shape}")

# Fill missing values
user_data = user_data.fillna("")
message_data = message_data.fillna("")

# Perform sentiment analysis
print("Performing sentiment analysis...")

def get_sentiment(text):
    """Calculate sentiment using TextBlob."""
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return TextBlob(text).sentiment.polarity

# Calculate sentiment for a sample of the messages (for speed)
sample_size = min(5000, len(message_data))
message_sample = message_data.sample(sample_size, random_state=42)
message_sample['sentiment_score'] = message_sample['message'].apply(get_sentiment)

print(f"Analyzed sentiment for {sample_size} messages")

# Generate results plots
print("Generating results...")

# Create success rate by sentiment bins
message_sample['sentiment_bin'] = pd.cut(message_sample['sentiment_score'], bins=5)
success_by_sentiment = message_sample.groupby('sentiment_bin')['success'].mean()

# Plot success rate by sentiment
plt.figure(figsize=(10, 6))
success_by_sentiment.plot(kind='bar')
plt.title('Invitation Success Rate by Message Sentiment')
plt.xlabel('Message Sentiment')
plt.ylabel('Success Rate')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'success_by_sentiment.png'))
plt.close()

# Create word count analysis
message_sample['word_count'] = message_sample['message'].apply(lambda x: len(str(x).split()))
message_sample['word_count_bin'] = pd.cut(message_sample['word_count'], bins=5)
success_by_word_count = message_sample.groupby('word_count_bin')['success'].mean()

# Plot success rate by word count
plt.figure(figsize=(10, 6))
success_by_word_count.plot(kind='bar')
plt.title('Invitation Success Rate by Message Word Count')
plt.xlabel('Word Count')
plt.ylabel('Success Rate')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'success_by_word_count.png'))
plt.close()

# Generate a confusion matrix (mock results)
# Generate mock predictions based on sentiment
message_sample['pred'] = (message_sample['sentiment_score'] > 0.2).astype(int)

# Create confusion matrix
cm = confusion_matrix(message_sample['success'], message_sample['pred'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Sample Results)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# Calculate evaluation metrics
tp = cm[1, 1]
fp = cm[0, 1]
fn = cm[1, 0]
tn = cm[0, 0]

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

mock_metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score
}

# Save metrics to CSV
pd.DataFrame([mock_metrics]).to_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'), index=False)

print("Results saved to:", OUTPUT_DIR)
print("Done!")
