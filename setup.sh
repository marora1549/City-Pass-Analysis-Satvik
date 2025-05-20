#!/bin/bash
# Setup script for city pass analysis project

# Create a virtual environment
echo "Creating virtual environment..."
python -m venv city_pass_env

# Activate the virtual environment
echo "Activating virtual environment..."
source city_pass_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create output directory
echo "Creating output directory..."
mkdir -p output

echo "Setup complete! Virtual environment 'city_pass_env' has been created."
echo "To activate the environment, run: source city_pass_env/bin/activate"
