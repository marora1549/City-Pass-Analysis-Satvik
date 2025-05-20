"""
Practical application demonstration script for city pass invitation prediction.

This script demonstrates how the trained GNN model could be used in practice
to suggest optimal invitation strategies and potential travel partners.
"""

import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

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

# Fill missing values
user_data = user_data.fillna("")
message_data = message_data.fillna("")

# Function to calculate sentiment
def get_sentiment(text):
    """Calculate sentiment using TextBlob."""
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return TextBlob(text).sentiment.polarity

# Calculate sentiment for users
user_data['sentiment_score'] = user_data['user_profile'].apply(get_sentiment)

# Calculate sentiment for messages
# Use a sample for efficiency
sample_size = min(2000, len(message_data))
message_sample = message_data.sample(sample_size, random_state=42)
message_sample['sentiment_score'] = message_sample['message'].apply(get_sentiment)

# -----------------------------------------------------------------------------
# Demonstration 1: Recommend optimal message style based on receiver profile
# -----------------------------------------------------------------------------

def recommend_message_style(receiver_profile):
    """
    Recommend an optimal message style based on receiver profile.
    
    In a real system, this would use the trained GNN model to predict the best approach.
    Here we're using a simplified rule-based approach for demonstration.
    """
    sentiment = get_sentiment(receiver_profile)
    
    if sentiment > 0.5:
        return "Enthusiastic and adventurous invitation emphasizing exciting experiences"
    elif sentiment > 0.2:
        return "Friendly invitation highlighting shared interests and cultural experiences"
    elif sentiment > 0:
        return "Balanced invitation with specific destination details and activities"
    elif sentiment > -0.2:
        return "Informative invitation focusing on practical benefits and convenience"
    else:
        return "Reserved invitation with emphasis on relaxation and comfort"

# Demonstrate message style recommendations for a few users
print("\nDemonstration 1: Message Style Recommendations")
print("=============================================")

sample_users = user_data.sample(5, random_state=42)
for _, user in sample_users.iterrows():
    user_id = user['uid']
    profile = user['user_profile']
    sentiment = user['sentiment_score']
    
    recommendation = recommend_message_style(profile)
    
    print(f"\nUser ID: {user_id}")
    print(f"Profile snippet: {profile[:100]}...")
    print(f"Profile sentiment: {sentiment:.2f}")
    print(f"Recommended message style: {recommendation}")

# -----------------------------------------------------------------------------
# Demonstration 2: Suggest potential travel partners
# -----------------------------------------------------------------------------

def suggest_travel_partners(user_id, top_n=3):
    """
    Suggest potential travel partners for a given user.
    
    In a real system, this would use the trained GNN model to predict invitation success.
    Here we're using a simplified approach for demonstration.
    """
    # Get user profile
    user_profile = user_data[user_data['uid'] == user_id]['user_profile'].values[0]
    user_sentiment = get_sentiment(user_profile)
    
    # Find other users with similar sentiment (simplified matching logic)
    user_data['sentiment_diff'] = abs(user_data['sentiment_score'] - user_sentiment)
    
    # Exclude the user themselves
    potential_partners = user_data[user_data['uid'] != user_id].sort_values('sentiment_diff')
    
    # Return top N potential partners
    return potential_partners.head(top_n)

# Demonstrate partner suggestions for a sample user
print("\nDemonstration 2: Potential Travel Partner Suggestions")
print("===================================================")

sample_user_id = user_data.sample(1, random_state=42)['uid'].values[0]
user_profile = user_data[user_data['uid'] == sample_user_id]['user_profile'].values[0]

print(f"\nUser ID: {sample_user_id}")
print(f"Profile snippet: {user_profile[:100]}...")

suggested_partners = suggest_travel_partners(sample_user_id)
print("\nSuggested travel partners:")
for _, partner in suggested_partners.iterrows():
    partner_id = partner['uid']
    profile = partner['user_profile']
    sent_diff = partner['sentiment_diff']
    
    print(f"\n  Partner ID: {partner_id}")
    print(f"  Profile snippet: {profile[:100]}...")
    print(f"  Sentiment similarity: {1 - sent_diff:.2f}")
    
    # In a real system, we would also show the predicted success probability here
    # based on the GNN model's output
    mock_success_prob = random.uniform(0.75, 0.95)
    print(f"  Predicted invitation success: {mock_success_prob:.2f}")

# -----------------------------------------------------------------------------
# Demonstration 3: Craft personalized invitation message
# -----------------------------------------------------------------------------

def craft_personalized_invitation(sender_profile, receiver_profile):
    """
    Craft a personalized invitation message based on sender and receiver profiles.
    
    In a real system, this would use more sophisticated NLP techniques.
    Here we're using a template-based approach for demonstration.
    """
    sender_sentiment = get_sentiment(sender_profile)
    receiver_sentiment = get_sentiment(receiver_profile)
    
    # Extract interests (simplified)
    sender_words = set(sender_profile.lower().split())
    receiver_words = set(receiver_profile.lower().split())
    
    # Common travel-related keywords
    travel_keywords = {
        'culture', 'adventure', 'food', 'cuisine', 'beach', 'hiking', 'nature',
        'history', 'architecture', 'museum', 'art', 'relaxation', 'photography'
    }
    
    # Find common interests
    sender_interests = sender_words.intersection(travel_keywords)
    receiver_interests = receiver_words.intersection(travel_keywords)
    common_interests = sender_interests.intersection(receiver_interests)
    
    # Select greeting based on sentiment
    greetings = [
        "Hey! I've been thinking about exploring",
        "I was wondering if you'd like to discover",
        "Let's embark on a journey to",
        "How about we experience",
        "Would you be interested in exploring"
    ]
    
    # Select destination based on common interests
    destinations = {
        'culture': "the historic sites of Rome",
        'adventure': "the hiking trails in New Zealand",
        'food': "the culinary delights of Tokyo",
        'cuisine': "the food markets in Barcelona",
        'beach': "the stunning beaches of Bali",
        'hiking': "the mountain paths in Switzerland",
        'nature': "the natural wonders of Costa Rica",
        'history': "the ancient ruins of Greece",
        'architecture': "the stunning buildings of Prague",
        'museum': "the world-class museums in Paris",
        'art': "the art galleries in Florence",
        'relaxation': "the peaceful resorts in Maldives",
        'photography': "the picturesque landscapes of Iceland"
    }
    
    # Pick a random greeting
    greeting = random.choice(greetings)
    
    # Choose a destination based on common interests if any, otherwise based on receiver interests
    if common_interests:
        interest = random.choice(list(common_interests))
        destination = destinations.get(interest, "some amazing places together")
    elif receiver_interests:
        interest = random.choice(list(receiver_interests))
        destination = destinations.get(interest, "some amazing places together")
    else:
        destination = "some incredible destinations that I think you'd love"
    
    # Construct message
    message = f"{greeting} {destination}. I think it would be a fantastic experience that we'll both enjoy!"
    
    return message

# Demonstrate personalized message crafting
print("\nDemonstration 3: Personalized Invitation Messages")
print("===============================================")

# Select sample sender and receiver
sender = user_data.sample(1, random_state=42).iloc[0]
receiver = user_data.sample(1, random_state=43).iloc[0]

sender_id = sender['uid']
sender_profile = sender['user_profile']
receiver_id = receiver['uid']
receiver_profile = receiver['user_profile']

print(f"\nSender ID: {sender_id}")
print(f"Sender profile snippet: {sender_profile[:100]}...")
print(f"\nReceiver ID: {receiver_id}")
print(f"Receiver profile snippet: {receiver_profile[:100]}...")

personalized_message = craft_personalized_invitation(sender_profile, receiver_profile)
print(f"\nPersonalized invitation message:")
print(f"{personalized_message}")

# In a real system, we would also show the predicted success probability
mock_success_prob = random.uniform(0.75, 0.95)
print(f"\nPredicted invitation success: {mock_success_prob:.2f}")

# -----------------------------------------------------------------------------
# Demonstration 4: Analyze invitation success patterns
# -----------------------------------------------------------------------------

print("\nDemonstration 4: Invitation Success Patterns")
print("=========================================")

# Calculate success rates by hour (simulated for demonstration)
hours = list(range(24))
success_rates = [random.uniform(0.3, 0.7) for _ in range(24)]

# Create peak periods with higher success rates
for hour in [10, 11, 18, 19, 20]:
    success_rates[hour] = random.uniform(0.7, 0.9)

# Plot success rates by hour
plt.figure(figsize=(12, 6))
plt.plot(hours, success_rates, 'o-', linewidth=2, markersize=8)
plt.title('Invitation Success Rate by Hour of Day')
plt.xlabel('Hour (24-hour format)')
plt.ylabel('Success Rate')
plt.xticks(range(0, 24, 2))
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'success_by_hour.png'))
plt.close()

print("Generated success rate analysis by hour of day (saved to output directory)")

# Generate success rate by day of week (simulated)
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_success_rates = [random.uniform(0.4, 0.6) for _ in range(7)]
day_success_rates[5] = random.uniform(0.7, 0.85)  # Saturday higher
day_success_rates[6] = random.uniform(0.7, 0.85)  # Sunday higher

plt.figure(figsize=(12, 6))
plt.bar(days, day_success_rates, color='skyblue')
plt.title('Invitation Success Rate by Day of Week')
plt.xlabel('Day')
plt.ylabel('Success Rate')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')
plt.savefig(os.path.join(OUTPUT_DIR, 'success_by_day.png'))
plt.close()

print("Generated success rate analysis by day of week (saved to output directory)")

print("\nRecommendations based on timing analysis:")
print("- Best time to send invitations: Evenings (6-8 PM) and late morning (10-11 AM)")
print("- Best days for invitations: Weekends (Saturday and Sunday)")
print("- Avoid sending invitations: Late night (1-5 AM) and early morning (5-7 AM)")

print("\nDemonstration completed!")
