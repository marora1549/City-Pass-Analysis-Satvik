#!/bin/bash
# Run script for city pass analysis project

# Activate the virtual environment
source city_pass_env/bin/activate

# Execute the main program
python src/main.py \
    --user_data data/user_data.csv \
    --message_data data/message_data.csv \
    --model_type gcn \
    --hidden_dim 64 \
    --sentiment_method textblob \
    --epochs 100 \
    --lr 0.001 \
    --early_stopping 10 \
    --seed 42 \
    --output_dir output

echo "Execution complete! Results saved to the 'output' directory."
