#!/bin/bash
# Step 1: Install VERL for GRPO training

set -e

echo "============================================"
echo "  Step 1: Installing VERL"
echo "============================================"
echo ""

# Check Python version
echo "[1/4] Checking Python version..."
python --version
echo ""

# Install core dependencies first
echo "[2/4] Installing core dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
echo ""

# Install VERL from GitHub (it's not on PyPI)
echo "[3/4] Installing VERL from GitHub..."
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
cd ..
echo ""

# Verify installation
echo "[4/4] Verifying VERL installation..."
python -c "import verl; print(f'✓ VERL installed successfully: version {verl.__version__}')" || echo "⚠ VERL import failed"
echo ""

echo "============================================"
echo "   Step 1 Complete!"
echo "============================================"
echo ""
# echo "Next step: Prepare your data for VERL"
# echo "Run: python step2_prepare_data.py"