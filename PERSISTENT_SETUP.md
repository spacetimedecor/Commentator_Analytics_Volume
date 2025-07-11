# Persistent Volume Setup Guide

This workspace is configured to use your 200GB persistent volume for all dependencies, caches, and models.

## 🎯 What's Configured

### Automatic Environment Activation
- Your shell (`.bashrc`) automatically loads the workspace environment
- All cache directories point to `/workspace/.cache/`
- Python virtual environment auto-activates

### Cache Locations (All on 200GB volume)
```
/workspace/.cache/pip/          # Python package cache
/workspace/.cache/huggingface/  # HuggingFace models & datasets
/workspace/.cache/torch/        # PyTorch models
/workspace/.cache/conda/        # Conda packages (if used)
/workspace/.cache/npm/          # Node.js packages (if used)
/workspace/tmp/                 # Temporary files
/workspace/models/              # Downloaded ML models
```

## 📦 Installing New Packages

### Python Packages
```bash
# These will automatically use workspace cache
pip install new-package
pip install -r requirements.txt
```

### Conda/Mamba (if you switch to conda)
```bash
# Will use /workspace/.cache/conda/pkgs
conda install new-package
mamba install new-package
```

### Node.js (if needed)
```bash
# Will use /workspace/.cache/npm
npm install package-name
```

## 🔄 Environment Activation Methods

### Method 1: Auto-activation (Recommended)
- Just open a new terminal - environment loads automatically
- Works for SSH sessions, VS Code terminals, etc.

### Method 2: Manual activation
```bash
source /workspace/activate_env.sh
```

### Method 3: Direct source
```bash
source /workspace/.workspace_env
source /workspace/venv/bin/activate
```

## ✅ Verification Commands

Check that everything uses workspace volume:
```bash
echo $PIP_CACHE_DIR          # Should show /workspace/.cache/pip
echo $HUGGINGFACE_HUB_CACHE  # Should show /workspace/.cache/huggingface
echo $TORCH_HOME             # Should show /workspace/.cache/torch
pip config list             # Should show workspace cache dir
```

## 🚫 What NOT to Do

❌ **Don't use system pip**: `sudo pip install` (bypasses our setup)  
❌ **Don't clear workspace cache**: `rm -rf /workspace/.cache/` (you'll lose 4.6GB of packages)  
❌ **Don't install to user directory**: `pip install --user` (goes to container filesystem)  

## 🛠️ Troubleshooting

### If packages install to wrong location:
```bash
# Re-activate environment
source /workspace/activate_env.sh

# Check pip config
pip config list

# Verify paths
which python
which pip
```

### If running out of container space:
```bash
# Clean temporary files
rm -rf /tmp/pip-*
rm -rf /root/.cache/*

# Check what's using space
df -h
du -sh /workspace/.cache/*
```

## 📁 Key Files

- `/workspace/.workspace_env` - Environment variables
- `/workspace/.pip/pip.conf` - Pip configuration  
- `/workspace/activate_env.sh` - Manual activation script
- `~/.bashrc` - Auto-activation setup
- `/workspace/requirements.txt` - Package list

This setup ensures all future dependencies automatically use your persistent 200GB volume! 🎉