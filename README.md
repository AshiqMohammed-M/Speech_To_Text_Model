# Speech to Text From Scratch

Real-time transcription that converts live audio into clean, noise-reduced text.

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
