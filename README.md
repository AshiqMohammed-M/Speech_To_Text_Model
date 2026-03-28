# Speech to Text From Scratch

Real-time transcription that converts live audio into clean, noise-reduced text.

---

## Installation

This project uses native audio decoding, so FFmpeg must be installed and available on PATH.

### Windows + uv
1. Install uv:
   - ```winget install AstralSoftware.UV``` or if you have python installed ```pip install uv```
2. Install FFmpeg:
   - ```winget install Gyan.FFmpeg```
3. Install project dependencies:
   - ```uv sync```

### Windows + pip
1. Create and activate a virtual environment:
   - ```python -m venv .venv
   .venv\Scripts\activate```
2. Upgrade the package manager:
   - ```python -m pip install --upgrade pip```
3. Install CPU PyTorch and TorchCodec from the PyTorch CPU index:
   - ```python -m pip install torch==2.11.0+cpu torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu```
4. Install remaining project dependencies:
   - ```python -m pip install -r requirements.txt```
5. Install FFmpeg:
   - ```winget install Gyan.FFmpeg```

### macOS + uv
1. Install uv:
   - ```brew install uv```
2. Install FFmpeg:
   - ```brew install ffmpeg```
3. Install project dependencies:
   - ```uv sync```

### macOS + pip
1. Create and activate a virtual environment:
   ``` python3 -m venv .venv
   source .venv/bin/activate```
2. Upgrade the package manager:
   - ```python -m pip install --upgrade pip```
3. Install CPU PyTorch and TorchCodec from the PyTorch CPU index:
   - ```python -m pip install torch==2.11.0+cpu torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu```
4. Install remaining project dependencies:
   - ```python -m pip install -r requirements.txt```
5. Install FFmpeg:
   - ```brew install ffmpeg```

### Linux + uv
1. Install uv:
   - ```curl -LsSf https://astral.sh/uv/install.sh | sh``` or if you have python installed ```pip install uv```
2. Install FFmpeg (Ubuntu or Debian):
   - ```sudo apt update && sudo apt install -y ffmpeg```
3. Install project dependencies:
   - ```uv sync```

### Linux + pip
1. Create and activate a virtual environment:
   - ```python3 -m venv .venv
   source .venv/bin/activate```
2. Upgrade the package manager:
   - ```python -m pip install --upgrade pip```
3. Install CPU PyTorch and TorchCodec from the PyTorch CPU index:
   - ```python -m pip install torch==2.11.0+cpu torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu```
4. Install remaining project dependencies:
   - ```python -m pip install -r requirements.txt```
5. Install FFmpeg (Ubuntu or Debian):
   - ``sudo apt update && sudo apt install -y ffmpeg``

---

## Features
- Real-time live audio capture and transcription.
- Noise-robust preprocessing pipeline.
- Deep learning architecture built end-to-end in PyTorch.
- Designed for research and production-ready adaptation.

---

## Architecture & Pipeline
1. **Convolution Layer** — Performs initial feature extraction from raw audio waveform.  
2. **Self-Attention (Transformer) Layer** — Models long-range temporal dependencies in audio sequences.  
3. **Residual Vector Quantizer** — Compresses representations while maintaining expressiveness.  
4. **Text Predictor** — Maps learned latent features to textual output via sequence generation.

---
