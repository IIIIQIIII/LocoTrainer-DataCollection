# Troubleshooting Guide

Common issues and solutions for LocoTrainer-DataCollection.

## Installation Issues

### transformers Version Incompatibility

**Problem**: LocoTrainer-4B fails to load with error about missing model architecture or incompatible transformers.

**Cause**: vLLM bundles an old version of transformers that doesn't support LocoTrainer-4B's architecture.

**Solution**:
```bash
# CRITICAL: Install vLLM first, THEN upgrade transformers
source .venv/bin/activate
uv pip install vllm
uv pip install transformers==5.2.0

# Verify versions
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
# Should output: transformers: 5.2.0
```

**Prevention**: The `setup_cloud.sh` script handles this automatically. If you installed manually, always upgrade transformers after vLLM.

---

## vLLM Deployment Issues

### Port Already in Use

**Problem**: `deploy_vllm_8gpu.sh` fails with "Address already in use" error.

**Solution**:
```bash
# Kill existing vLLM processes
pkill -f "vllm serve"

# Or check what's using the port
lsof -i :8000

# Then redeploy
./deploy_vllm_8gpu.sh
```

---

### Out of Memory (OOM) Errors

**Problem**: vLLM crashes with CUDA OOM or "Failed to allocate memory".

**Solution 1 - Reduce max context**:
```bash
# Edit deploy_vllm_8gpu.sh, change:
--max-model-len 131072  # to lower value
--max-model-len 65536   # Try 64k context instead
```

**Solution 2 - Reduce GPU memory utilization**:
```bash
# Change in deploy_vllm_8gpu.sh:
--gpu-memory-utilization 0.95  # to
--gpu-memory-utilization 0.85  # or 0.80
```

**Solution 3 - Reduce batch size**:
```bash
# Add to vLLM launch command:
--max-num-seqs 1  # Process one request at a time per GPU
```

---

### Model Download Fails

**Problem**: vLLM hangs or fails downloading LocoTrainer-4B from HuggingFace.

**Solution**:
```bash
# Pre-download model manually
source .venv/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('LocoreMind/LocoTrainer-4B', cache_dir='~/.cache/huggingface')"

# Then vLLM will use cached model
./deploy_vllm_8gpu.sh
```

**Alternative - Use HF mirror** (for China):
```bash
export HF_ENDPOINT=https://hf-mirror.com
./deploy_vllm_8gpu.sh
```

---

## Collection Issues

### Queries Timing Out

**Problem**: Queries consistently fail with timeout or max_turns reached.

**Solution**:
```bash
# Increase max turns and tokens
python batch_collect.py \
    --max-turns 50 \
    --max-tokens 20480
```

---

### Low Success Rate

**Problem**: Many queries fail with errors in collection_summary.json.

**Check 1 - vLLM health**:
```bash
# Test each vLLM instance
for port in {8000..8007}; do
    echo "Testing port $port:"
    curl http://localhost:$port/v1/models
done
```

**Check 2 - Review error logs**:
```bash
# Find failed queries
ls trajectories/*/error.json

# Check specific error
cat trajectories/msswift_0042/error.json
```

**Check 3 - ms-swift path**:
```bash
# Verify ms-swift exists
ls -la /workspace/ms-swift

# If missing, clone manually
git clone https://github.com/modelscope/ms-swift.git /workspace/ms-swift
```

---

### Slow Generation Speed

**Problem**: Each query takes 20+ minutes instead of 8-12 minutes.

**Solution 1 - Install Flash Attention**:
```bash
source .venv/bin/activate
pip install flash-attn --no-build-isolation
# Restart vLLM after installation
```

**Solution 2 - Check GPU utilization**:
```bash
nvidia-smi
# If GPU utilization is low, you may have bottleneck elsewhere
```

**Solution 3 - Reduce temperature** (faster but less diverse):
```bash
# Edit .env
LOCOTRAINER_TEMPERATURE=0.3  # Down from 0.7
```

---

## Environment Issues

### Python Version Mismatch

**Problem**: vLLM or transformers fail with Python version error.

**Solution**:
```bash
python --version  # Check current version
# LocoTrainer-4B requires Python 3.10+

# If using older Python, install 3.10+
sudo apt-get install python3.10 python3.10-venv
uv venv --python python3.10
```

---

### CUDA Version Incompatibility

**Problem**: vLLM fails with CUDA errors.

**Check CUDA version**:
```bash
nvidia-smi | grep "CUDA Version"
# vLLM requires CUDA 12.1+
```

**Solution**:
```bash
# Update NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-535  # or latest

# Reboot required
sudo reboot
```

---

## Monitoring Issues

### Can't See Progress

**Problem**: Collection is running but no output/progress visible.

**Solution**:
```bash
# Check active Python processes
ps aux | grep batch_collect

# Count completed trajectories
ls trajectories/*/trajectory.json | wc -l

# View latest logs
tail -f vllm_gpu0.log

# Check collection summary (updates live)
watch -n 10 'cat trajectories/collection_summary.json 2>/dev/null | jq ".successful, .failed"'
```

---

## Resume Failed Collection

**Problem**: Collection stopped midway, need to resume.

**Solution**:
```bash
# Check how many completed
ls trajectories/*/trajectory.json | wc -l
# Example output: 234

# Resume from that index
python batch_collect.py --start-idx 234
```

---

## GitHub Upload Issues

### Large File Warnings

**Problem**: Git warns about large files when pushing trajectories.

**Solution**:
```bash
# Don't commit trajectories to Git!
# Upload to HuggingFace instead:

pip install huggingface_hub[cli]
huggingface-cli login

# Create dataset repo
huggingface-cli repo create msswift-locotrainer-trajectories --type dataset

# Upload
huggingface-cli upload IIIIQIIII/msswift-locotrainer-trajectories ./trajectories
```

---

## Performance Optimization

### Maximize Throughput

**Optimal configuration for 8x H100**:

```bash
# 1. Use separate vLLM instance per GPU (already default)
./deploy_vllm_8gpu.sh

# 2. Increase workers per GPU if queries are short
python batch_collect.py --workers-per-gpu 2  # Caution: may cause OOM

# 3. Disable verbose logging
python batch_collect.py --quiet

# 4. Use faster sampling
export LOCOTRAINER_TEMPERATURE=0.3
export LOCOTRAINER_TOP_P=0.95
```

---

## Getting Help

If your issue isn't listed here:

1. **Check logs**:
   - `vllm_gpu*.log` - vLLM server logs
   - `trajectories/*/error.json` - Per-query errors
   - `trajectories/collection_summary.json` - Overall stats

2. **Create GitHub issue**: https://github.com/IIIIQIIII/LocoTrainer-DataCollection/issues
   - Include error messages
   - Include environment info (GPU model, CUDA version, Python version)
   - Include relevant log excerpts

3. **Original LocoTrainer issues**: https://github.com/LocoreMind/LocoTrainer/issues
