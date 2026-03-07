import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------
# CONFIG
# ------------------

block_size = 128
n_embd = 192
n_head = 4
n_layer = 4
dropout = 0.2
# Use CPU for deployment (MPS only on Apple Silicon, CUDA needs GPU)
device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)

# ------------------
# LOAD VOCAB
# ------------------

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "input.txt")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Vocab size:", vocab_size)  # debug check

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    # Use index 0 for unknown chars (avoids KeyError on deployment)
    return [stoi.get(c, 0) for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # compute attention scores
        wei = q @ k.transpose(-2, -1)  # (B,T,T)
        wei = wei * (self.head_size ** -0.5)  # scale

        # apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B,T,head_size)
        out = wei @ v      # (B,T,head_size)

        return out
    
class MultiHeadAttention(nn.Module):
        """Multiple heads of self-attention in parallel"""

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList(
                [Head(head_size) for _ in range(num_heads)]
            )
            self.proj = nn.Linear(n_embd,n_embd)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            # Concatenate outputs of all heads along the channel dimension
            out = torch.cat([h(x) for h in self.heads], dim=-1) 
            out = self.proj(out)
            return out

class FeedFoward(nn.Module):
        """A simple linear layer followed by a non-linearity"""

        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),  # expand
                nn.ReLU(),
                nn.Linear(4 *  n_embd, n_embd),
                nn.Dropout (dropout),
            )

        def forward(self, x):
            return self.net(x)
        

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x+self.sa(self.ln1(x))
        x = x+self.ffwd(self.ln2(x))
        return x


# -----------------------------
# Language Model with Attention
# -----------------------------


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ------------------
# LOAD MODEL
# ------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tiny_transformer.pth")

print("Model path:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

model = BigramLanguageModel(vocab_size)
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device)
model.eval()


# ------------------
# GENERATION FUNCTION
# ------------------

def generate_text(prompt: str, max_tokens: int = 100):
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_tokens)[0].tolist()
    return decode(out)