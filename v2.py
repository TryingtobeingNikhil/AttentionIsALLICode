import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters (FAST EXPERIMENT MODE)

batch_size = 32        # smaller batch
block_size = 128       # smaller context window
max_iters = 2000       # fewer steps
eval_interval = 200
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # M2 GPU
eval_iters = 100

n_embd = 192           # smaller embedding
n_head = 4             # fewer heads
n_layer = 4            # fewer layers
dropout = 0.2

torch.manual_seed(1337)

# load dataset
if not os.path.exists("input.txt"):
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open("input.txt", "w") as f:
        f.write(requests.get(url).text)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------
# Self-Attention Head
# -----------------------------

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

    def __init__(self):
        super().__init__()

        # token + positional embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # stacked transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head = n_head) for _ in range(n_layer)]
        )

        # final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # final language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]  # crop context
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]  # last time step
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# -----------------------------
# Training
# -----------------------------

model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# -----------------------------
# Generate text
# -----------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))