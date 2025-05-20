# City Pass Sentiment Analysis & Invitation Success Prediction

## Project Overview
This project aims to build a predictive model using Graph Neural Networks (GNNs) to predict the success of invitations between users of a city pass application. The model will leverage sentiment analysis of user profiles and message content to identify patterns that lead to successful invitations.

## Data Sources
1. **user_data.csv**: Contains user profiles with sentiment about travel experiences
2. **message_data.csv**: Contains invitation messages between users, with success/failure indicators

## Project Phases

### 1. Data Exploration & Preprocessing
- Analyze user profiles for sentiment patterns
- Analyze message content and success rates
- Clean and preprocess text data
- Extract relevant features for sentiment analysis
- Create a graph structure representing user connections

### 2. Sentiment Analysis Implementation
- Develop sentiment analysis approaches for both user profiles and messages
- Compare traditional ML methods with deep learning approaches
- Evaluate sentiment analysis performance
- Extract sentiment features to augment the graph structure

### 3. Graph Neural Network Design
- Design appropriate GNN architecture
- Implement node feature representation
- Define edge features based on messages and relationships
- Implement training and evaluation pipeline
- Tune hyperparameters for optimal performance

### 4. Model Evaluation & Analysis
- Evaluate model performance metrics
- Analyze prediction patterns and failure cases
- Identify key factors contributing to invitation success
- Compare with baseline models

### 5. Application & Recommendations
- Develop strategies to leverage the model for promoting travel opportunities
- Design methods to test effectiveness of these strategies
- Create visualization of prediction results
- Prepare comprehensive documentation and reports

## Technical Approach

### Sentiment Analysis Techniques
- Rule-based approaches (lexicon-based)
- Machine learning approaches (feature engineering + classification)
- Deep learning approaches (transformers, BERT, etc.)

### Graph Neural Network Architecture Options
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- GraphSAGE
- Graph Isomorphism Networks (GIN)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Graph-specific metrics
- Cross-validation strategies

## Expected Deliverables
1. Well-documented Python code for all implementations
2. Trained GNN model for invitation success prediction
3. Comprehensive analysis report of results
4. Testing modules for all components
5. Recommendations for practical application
