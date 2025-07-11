#!/bin/bash

# Activate workspace environment for audio processing models
# This script configures the environment to use the 200GB workspace volume
# for caching models and pip packages instead of the smaller container filesystem

# Load persistent environment configuration
source /workspace/.workspace_env

# Activate the Python virtual environment
source /workspace/venv/bin/activate

# Set pip config location
export PIP_CONFIG_FILE=/workspace/.pip/pip.conf

echo "🚀 Workspace environment activated!"
echo "- Python virtual environment: activated"
echo "- Pip cache: /workspace/.cache/pip"
echo "- Model cache: /workspace/models"
echo "- HuggingFace cache: /workspace/.cache/huggingface"
echo "- PyTorch cache: /workspace/.cache/torch"
echo "- All future installs will use workspace volume"
echo ""
echo "📦 To install new packages:"
echo "  pip install package-name"
echo ""
echo "🤖 To run the model download script:"
echo "  python scripts/download_models.py"