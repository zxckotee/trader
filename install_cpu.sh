#!/bin/bash

# Installation script for CPU-only version of trader_ml

echo "Installing CPU-only version of trader_ml dependencies..."

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install transformers>=4.30.0
pip install peft>=0.4.0
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install requests>=2.31.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install scikit-learn>=1.3.0
pip install tqdm>=4.65.0
pip install python-dotenv>=1.0.0
pip install ccxt>=4.0.0
pip install ta>=0.10.2

echo "Installation completed!"
echo ""
echo "To verify installation, run:"
echo "python -c \"import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')\""
