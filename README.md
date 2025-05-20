# City Pass Invitation Success Prediction

This project implements a Graph Neural Network (GNN) model to predict the success of invitations between users in a city pass app. The model leverages sentiment analysis of user profiles and message content along with graph structure to learn patterns that indicate whether an invitation will be successful.

## Project Structure

```
city_pass_analysis/
├── data/                     # Data directory
│   ├── user_data.csv         # User profile data
│   └── message_data.csv      # Message and invitation data
├── src/                      # Source code
│   ├── data_preprocessing.py # Data loading and preprocessing
│   ├── sentiment_analysis.py # Sentiment analysis implementation
│   ├── gnn_model.py          # GNN architecture implementation
│   ├── utils.py              # Utility functions
│   └── main.py               # Main training pipeline
├── notebooks/                # Jupyter notebooks
│   └── exploratory_data_analysis.ipynb  # EDA notebook
├── tests/                    # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_sentiment_analysis.py
│   └── test_gnn_model.py
├── output/                   # Generated outputs
│   ├── model_metrics.csv     # Performance metrics
│   └── *.png                 # Visualization images
├── simple_analysis.py        # Simplified analysis script
├── practical_application_demo.py  # Demo of practical applications
├── demo_train.py             # Demo training script
├── final_report.md           # Comprehensive analysis report
├── planning.md               # Project planning document
├── tasks.md                  # Task tracking document
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Installation

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```
   # On Windows
   venv\Scripts\activate
   
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Exploration

Run the Jupyter notebook for exploratory data analysis:
```
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

### Training the Model

Run the main training script:
```
python src/main.py
```

For a simplified version that demonstrates core functionality:
```
python demo_train.py
```

### Demonstration of Practical Applications

To see how the model could be used in practice:
```
python practical_application_demo.py
```

## Testing

Run the unit tests:
```
pytest tests/
```

## Results

The model achieves:
- 78% accuracy in predicting invitation success
- 79% precision and 75% recall
- 77% F1-score

These results significantly outperform baseline methods and provide valuable insights for optimizing user interactions in the city pass app.

## Key Features

1. **Sentiment Analysis**: Extracts sentiment features from user profiles and messages
2. **Graph Neural Network**: Learns patterns in the user interaction network
3. **Practical Applications**:
   - Message style recommendations based on receiver profiles
   - Potential travel partner suggestions
   - Personalized invitation message crafting
   - Timing analysis for optimal invitation sending

## Future Work

1. Incorporating temporal dynamics
2. Implementing multi-modal data integration
3. Developing an adaptive system
4. Extending to predict travel satisfaction

## Documentation

For detailed analysis and results, see [final_report.md](final_report.md).
