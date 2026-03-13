import os
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters

batch_size = 32        # how many independent sequences will we process in parallel?
block_size = 8         # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# load dataset


if not os.path.exists("input.txt"):
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open("input.txt", "w") as f:
        f.write(requests.get(url).text)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# character ↔ integer mapping
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -------------------
# data loader
# -------------------
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

# -------------------
# Bigram Model
# -------------------
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]          # focus only on last time step
            probs = F.softmax(logits, dim=-1)  # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat((idx, idx_next), dim=1)  # append
        return idx
        
# initialize model

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop

for iter in range(max_iters):

    # evaluate loss periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -------------------
# generate text
# -------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
