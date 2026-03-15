#!/bin/bash
# Complete setup script for cloud instance (Ubuntu 22.04+ with 8x H100)

set -e

echo "🚀 LocoTrainer Data Collection - Cloud Setup"
echo "============================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Please don't run as root. Run as regular user with sudo access."
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y git curl build-essential

# Verify CUDA
echo ""
echo "🔍 Checking NVIDIA GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $GPU_COUNT GPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$GPU_COUNT" -lt 8 ]; then
    echo "⚠️  Warning: Expected 8 GPUs, found $GPU_COUNT. Proceeding anyway..."
fi

# Install uv (Python package manager)
echo ""
echo "📦 Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi
uv --version

# Install Python 3.10+
echo ""
echo "🐍 Setting up Python environment..."
cd "$(dirname "$0")"
uv venv
source .venv/bin/activate

# Install project dependencies
echo ""
echo "📦 Installing LocoTrainer dependencies..."
uv pip install -e .

# Install vLLM (IMPORTANT: vLLM must be installed before transformers upgrade)
echo ""
echo "📦 Installing vLLM..."
uv pip install vllm

# Critical fix: vLLM bundles an old transformers that can't run LocoTrainer-4B
echo ""
echo "⚠️  Upgrading transformers to 5.2.0 (required for LocoTrainer-4B)..."
uv pip install transformers==5.2.0

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "from locotrainer import Agent, Config; print('LocoTrainer imported successfully')"

# Download LocoTrainer-4B model (optional pre-download)
echo ""
read -p "📥 Pre-download LocoTrainer-4B model? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading model (this may take several minutes)..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('LocoreMind/LocoTrainer-4B')"
    echo "✓ Model downloaded"
fi

# Clone ms-swift if needed
echo ""
MS_SWIFT_PATH="/workspace/ms-swift"
if [ ! -d "$MS_SWIFT_PATH" ]; then
    echo "📥 Cloning ms-swift repository..."
    sudo mkdir -p /workspace
    sudo chown $(whoami):$(whoami) /workspace
    git clone https://github.com/modelscope/ms-swift.git $MS_SWIFT_PATH
    echo "✓ ms-swift cloned to $MS_SWIFT_PATH"
else
    echo "✓ ms-swift already exists at $MS_SWIFT_PATH"
fi

# Create .env if not exists
if [ ! -f .env ]; then
    echo ""
    echo "⚙️  Creating .env configuration..."
    cat > .env << 'EOF'
# vLLM endpoints (8 parallel instances)
LOCOTRAINER_BASE_URL=http://localhost:8000/v1
LOCOTRAINER_MODEL=LocoreMind/LocoTrainer-4B
LOCOTRAINER_API_KEY=local

# Agent settings for 200k context
LOCOTRAINER_MAX_TURNS=40
LOCOTRAINER_MAX_TOKENS=16384

# Paths
LOCOTRAINER_CODEBASE=/workspace/ms-swift
LOCOTRAINER_OUTPUT_DIR=./trajectories
EOF
    echo "✓ .env created (edit if needed)"
fi

# Make scripts executable
chmod +x deploy_vllm_8gpu.sh
chmod +x batch_collect.py

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Deploy vLLM: ./deploy_vllm_8gpu.sh"
echo "  3. Start collection: python batch_collect.py --queries data/msswift_queries_500.json"
echo ""
echo "Quick test:"
echo "  python batch_collect.py --end-idx 5  # Test with first 5 queries"
echo ""
echo "Monitor:"
echo "  watch -n 1 nvidia-smi  # GPU usage"
echo "  tail -f vllm_gpu0.log  # vLLM logs"
echo ""
