#!/bin/bash

PYTHON=python

echo " Upgrading pip..."
$PYTHON -m pip install --upgrade pip -q

echo " Installing required Python packages..."
$PYTHON -m pip install --force-reinstall --no-cache-dir tqdm numba numpy torch more-itertools -q

echo " Installing Rust (required for tiktoken)..."
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

echo " Installing tiktoken..."
$PYTHON -m pip install --no-cache-dir tiktoken==0.3.3

echo " Installing whisper-at..."
$PYTHON -m pip install --no-deps whisper-at -q

# Optional: Install ffmpeg if you have sudo
if command -v sudo &> /dev/null; then
    echo " Installing ffmpeg..."
    sudo apt-get update -q
    sudo apt-get install -y ffmpeg -q
else
    echo " Skipping ffmpeg install — 'sudo' not available"
fi

echo " Setup complete! All dependencies installed."
