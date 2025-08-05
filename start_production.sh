#!/bin/bash
# Production startup script for SISL RAG API

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export RAG_ENV="production"

# Activate virtual environment (if using one)
# source /path/to/your/venv/bin/activate

# Start the API with production settings
python api.py
