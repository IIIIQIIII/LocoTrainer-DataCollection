# Quick Start Guide

Get started collecting MS-SWIFT trajectories in under 5 minutes.

## Prerequisites

- **Hardware**: Cloud instance with 8x H100 GPUs (80GB each)
- **OS**: Ubuntu 22.04+ with NVIDIA drivers installed
- **Network**: Fast internet for model download (~15GB)

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone https://github.com/IIIIQIIII/LocoTrainer-DataCollection.git
cd LocoTrainer-DataCollection
```

### 2. Run Automated Setup

```bash
chmod +x setup_cloud.sh
./setup_cloud.sh
```

This will:
- ✓ Install Python dependencies (uv, venv)
- ✓ Install vLLM for GPU inference
- ✓ Clone ms-swift repository
- ✓ Create `.env` configuration
- ✓ Download LocoTrainer-4B model (optional)

### 3. Activate Environment

```bash
source .venv/bin/activate
```

### 4. Deploy vLLM (8 parallel instances)

```bash
./deploy_vllm_8gpu.sh
```

Wait ~30s for model loading. You should see:
```
✓ GPU 0 (port 8000): OK
✓ GPU 1 (port 8001): OK
...
✓ GPU 7 (port 8007): OK
```

### 5. Start Collection

**Test run (5 queries, ~1 hour):**
```bash
python batch_collect.py --end-idx 5
```

**Full run (500 queries, ~8-10 hours):**
```bash
python batch_collect.py
```

## Monitoring

Open separate terminals to monitor:

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Collection progress
watch -n 5 'ls trajectories/*/trajectory.json | wc -l'

# vLLM logs
tail -f vllm_gpu0.log
```

## Results

Trajectories are saved to `./trajectories/`:

```
trajectories/
├── msswift_0001/
│   ├── trajectory.json    # Full conversation
│   └── output.md          # Analysis report
├── msswift_0002/
│   └── trajectory.json
...
└── collection_summary.json  # Overall stats
```

## Troubleshooting

**vLLM won't start:**
```bash
# Check GPU availability
nvidia-smi

# Check logs
cat vllm_gpu0.log
```

**Out of memory:**
```bash
# Reduce max context length
export LOCOTRAINER_MAX_TOKENS=8192
```

**Slow generation:**
```bash
# Install Flash Attention
pip install flash-attn --no-build-isolation
```

## Next Steps

After collection completes:
1. Review `collection_summary.json` for stats
2. Upload to HuggingFace: `huggingface-cli upload ...`
3. Use trajectories for model distillation

## Support

- GitHub Issues: https://github.com/IIIIQIIII/LocoTrainer-DataCollection/issues
- Original LocoTrainer: https://github.com/LocoreMind/LocoTrainer
