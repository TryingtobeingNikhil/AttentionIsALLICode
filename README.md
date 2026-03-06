# Transformer Architecture - Complete PyTorch Implementation from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Complete implementation of the Transformer architecture from the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) built entirely from first principles using PyTorch. **No pre-built transformer layers used** - every component implemented from scratch to demonstrate deep understanding of the architecture.

## 🎯 Overview

This project implements every component of the Transformer architecture without relying on PyTorch's built-in transformer modules, including:

- ✅ **Multi-head self-attention mechanisms** - Scaled dot-product attention with parallel heads
- ✅ **Positional encoding** - Sine-cosine encoding for sequence position information
- ✅ **Encoder-decoder architecture** - Complete 6-layer encoder and decoder stacks
- ✅ **Layer normalization and residual connections** - Stabilization techniques
- ✅ **Masked attention** - For autoregressive generation in decoder
- ✅ **Feed-forward networks** - Position-wise fully connected layers

## 🏗️ Architecture Details

```text
Transformer Model
│
├── Input Embedding + Positional Encoding
│
├── Encoder (6 layers)
│   ├── Multi-Head Self-Attention
│   │   ├── Query, Key, Value Projections
│   │   ├── Scaled Dot-Product Attention
│   │   └── Concatenate Heads
│   ├── Add & Norm (Residual Connection)
│   ├── Feed-Forward Network
│   │   ├── Linear (d_model → d_ff)
│   │   ├── ReLU Activation
│   │   └── Linear (d_ff → d_model)
│   └── Add & Norm (Residual Connection)
│
└── Decoder (6 layers)
├── Masked Multi-Head Self-Attention
├── Add & Norm
├── Multi-Head Cross-Attention (Encoder-Decoder)
├── Add & Norm
├── Feed-Forward Network
├── Add & Norm
└── Linear + Softmax (Output Projection)
```

