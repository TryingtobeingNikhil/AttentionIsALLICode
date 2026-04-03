# Transformer Architecture - Complete PyTorch Implementation from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Complete implementation of the Transformer architecture from the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al.,2017) built entirely from first principles using PyTorch. **No pre-built transformer layers used**  every component implemented from scratch to demonstrate deep understanding of the architecture.

## 🎯Overview

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

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/TryingtobeingNikhil/AttentionIsALLICode.git
cd AttentionIsALLICode

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

AttentionIsALLICode/
├── transformer.py              # Main Transformer model
├── attention.py                # Multi-head attention implementation
├── positional_encoding.py      # Positional encoding (sine-cosine)
├── encoder.py                  # Encoder block and stack
├── decoder.py                  # Decoder block and stack
├── feedforward.py              # Position-wise feed-forward network
├── layers.py                   # Layer normalization and utilities
├── train.py                    # Training script (example)
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file

##  Key Components Explained

### 1. Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    """
    Implements scaled dot-product attention with multiple parallel heads.
    
    Process:
    1. Linear projections for Q, K, V (d_model → d_model)
    2. Split into num_heads (d_model → num_heads × d_k)
    3. Apply scaled dot-product attention per head
    4. Concatenate heads
    5. Final linear projection
    
    Attention formula: Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

### 2. Positional Encoding
```python
class PositionalEncoding(nn.Module):
    """
    Injects information about position in the sequence.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
    - pos is the position in the sequence
    - i is the dimension
    """
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
```

### 3. Encoder Layer
```python
class EncoderLayer(nn.Module):
    """
    Single encoder layer consisting of:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Position-wise feed-forward network
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
```

### 4. Decoder Layer
```python
class DecoderLayer(nn.Module):
    """
    Single decoder layer consisting of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention (with encoder output)
    4. Add & Norm
    5. Position-wise feed-forward network
    6. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
```

## 📊 Model Parameters

Default configuration (matching original paper):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 512 | Model dimension |
| `num_heads` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of encoder/decoder layers |
| `d_ff` | 2048 | Feed-forward network dimension |
| `dropout` | 0.1 | Dropout probability |
| `max_seq_length` | 100 | Maximum sequence length |

**Total Parameters:** ~65M (varies with vocabulary size)

## 🎯 Training Example
```python
import torch.optim as optim
from torch.nn import CrossEntropyLoss

# Initialize model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=100,
    dropout=0.1
)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
criterion = CrossEntropyLoss(ignore_index=0)  # Ignore padding token

# Training loop
model.train()
for epoch in range(num_epochs):
    for src, tgt in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt[:, :-1])  # Teacher forcing
        
        # Calculate loss
        loss = criterion(
            output.reshape(-1, tgt_vocab_size),
            tgt[:, 1:].reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 🔍 Implementation Highlights

### Scaled Dot-Product Attention
```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute attention weights and apply to values.
    
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (for padding or future positions)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
```

### Masking for Autoregressive Generation
```python
def generate_square_subsequent_mask(size):
    """
    Generate mask to prevent attention to future positions.
    Used in decoder self-attention.
    
    Returns upper triangular matrix of -inf, with diagonal and below as 0.
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

## 🎓 Learning Resources

This implementation closely follows the original paper:

- 📄 **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** - Vaswani et al., 2017 (Original paper)
- 📚 **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** - Jay Alammar (Visual guide)
- 📖 **[Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)** - Harvard NLP (Line-by-line)
- 🎥 **[Attention Mechanism](https://www.youtube.com/watch?v=iDulhoQ2pro)** - StatQuest (Video explanation)

## 🛠️ Built With

- **PyTorch 2.0+** - Deep learning framework
- **NumPy** - Numerical computations
- **Math** - Mathematical operations
- **Python 3.8+** - Programming language

## 📝 Implementation Notes

- ✅ **No pre-built transformer layers** - Every component built from scratch
- ✅ **Batched operations** - Efficient parallel processing
- ✅ **Variable sequence lengths** - Supports dynamic input sizes
- ✅ **Training and inference modes** - Includes both forward and generation
- ✅ **Clean, readable code** - Extensive comments and documentation
- ✅ **Modular design** - Easy to understand and modify individual components

## 🧪 Testing
```python
# Test multi-head attention
from attention import MultiHeadAttention

attn = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(32, 50, 512)  # (batch, seq_len, d_model)
output, weights = attn(x, x, x)

print(f"Output shape: {output.shape}")  # [32, 50, 512]
print(f"Attention weights shape: {weights.shape}")  # [32, 8, 50, 50]
```

## 🚧 Future Improvements

- [ ] Add beam search for inference
- [ ] Implement label smoothing
- [ ] Add learning rate scheduling (warmup)
- [ ] Support for relative positional encoding
- [ ] Pre-training on large corpus
- [ ] Multi-GPU training support
- [ ] ONNX export for deployment

## 🤝 Acknowledgments

Based on the groundbreaking paper:
> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need**. In Advances in Neural Information Processing Systems 30 (NIPS 2017).

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Nikhil Mourya**
- GitHub: [@TryingtobeingNikhil](https://github.com/TryingtobeingNikhil)
- LinkedIn: [nikhil-mourya](https://linkedin.com/in/nikhil-mourya-36913a300)
- Email: tsmftxnikhil14@gmail.com

---

⭐ **Star this repository if you found it helpful!**

💡 **Questions or suggestions?** Open an issue or reach out!

---
