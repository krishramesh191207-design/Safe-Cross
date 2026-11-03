#!/usr/bin/env bash
# One-time setup on Raspberry Pi (64-bit Raspberry Pi OS Bookworm, Pi 4 or Pi 5).
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU-only torch wheel for aarch64 (ExecuTorch's Python runtime depends on torch tensors)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install executorch numpy opencv-python-headless
python -c "from executorch.runtime import Runtime; print('ExecuTorch runtime OK')"
