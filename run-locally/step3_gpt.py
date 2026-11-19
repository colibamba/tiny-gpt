"""
STEP 3: A tiny GPT.

The bigram model only sees ONE character. A GPT sees the last `block_size`
characters (128 by default) and uses ATTENTION to decide which of them
matter for predicting the next one. Attention is the key idea behind every
modern AI model, from this one to Claude. (GPT = Generative Pre-trained
Transformer; "Transformer" is the name for this attention-based design.)

The structure, from input to output:

    characters -> token embedding + position embedding
               -> n_layer x Block:
                      attention   (each position looks back at earlier ones)
                      MLP         (a small neural network applied to each position)
               -> final layer: scores for every possible next character

This is the same recipe as Llama and Qwen, just far smaller: about
800,000 parameters by default, vs 3 billion for Llama 3.2.

Run with defaults:
    py step3_gpt.py

Change the size or training (see README for experiments):
    py step3_gpt.py --name big --n_layer 6 --n_embd 256 --max_iters 5000

Generate text from a trained model:
    py step3_gpt.py --sample --name default --prompt "ROMEO:" --temperature 0.8

Each training run saves to runs/<name>/:
    log.csv      train and val loss over time (plot with plot_runs.py)
    model.pt     the trained model (the version with the best val loss)
    samples.txt  text written by the model as it trained
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Tokenizer, load_text, split

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--name", default="default", help="name of this run (folder in runs/)")
# Model size
p.add_argument("--n_layer", type=int, default=4, help="number of blocks stacked")
p.add_argument("--n_head", type=int, default=4, help="attention heads per block")
p.add_argument("--n_embd", type=int, default=128, help="size of each character's vector")
p.add_argument("--block_size", type=int, default=128, help="how many characters it can look back")
p.add_argument("--dropout", type=float, default=0.1, help="randomly switch off this fraction during training")
# Training
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--max_iters", type=int, default=3000, help="training steps")
p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--train_chars", type=int, default=0,
               help="use only this many characters of training text (0 = all). Small = easy to overfit")
# Sampling
p.add_argument("--sample", action="store_true", help="generate text from a saved model instead of training")
p.add_argument("--prompt", default="\n")
p.add_argument("--length", type=int, default=500)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
run_dir = Path("runs") / args.name


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """
    Each position asks "which earlier characters are relevant to me?" and
    takes a weighted mix of them.

    Every character produces three vectors:
      query: what am I looking for?
      key:   what do I contain?
      value: what do I pass on if someone attends to me?
    The match between one position's query and another's key decides how
    much attention it pays. "Causal" means a position may only look
    BACKWARDS: it can't peek at the character it's trying to predict.

    "Heads": attention runs several times in parallel (n_head), so different
    heads can track different things, e.g. one follows the current word,
    another the speaker's name.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)   # makes query, key, value at once
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        # Lower-triangular mask: 1 = allowed to look, 0 = the future (blocked)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape                          # batch, time (positions), channels
        q, k, v = self.qkv(x).split(C, dim=2)
        # Split into heads: (B, n_head, T, head_size)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)          # how well each query matches each key
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # block the future
        att = F.softmax(att, dim=-1)                              # scores -> weights that add to 1
        att = self.dropout(att)
        out = att @ v                                             # weighted mix of values
        out = out.transpose(1, 2).contiguous().view(B, T, C)      # stick the heads back together
        return self.dropout(self.proj(out))


class Block(nn.Module):
    """One layer: attention (gather information) then MLP (think about it)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # "x + ..." is a residual connection: each layer ADDS to what's there
        # rather than replacing it. It makes deep networks much easier to train.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)    # what each character is
        self.pos_emb = nn.Embedding(block_size, n_embd)    # where it is in the sequence
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)          # scores for the next character

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, length, temperature=1.0):
        """
        Write text one character at a time, feeding each choice back in.

        Temperature controls randomness:
          below 1 -> safer, more predictable (and more repetitive)
          above 1 -> more surprising (and more mistakes)
        It's the same "temperature" setting you fixed at 0 in your eval.
        """
        for _ in range(length):
            context = idx[:, -self.block_size:]       # it can only see block_size characters back
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
text = load_text()
tok = Tokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
train_data, val_data = split(data)
if args.train_chars:
    train_data = train_data[: args.train_chars]


def get_batch(source):
    """Random chunks of block_size characters. y is x shifted by one: the next character at every position."""
    i = torch.randint(len(source) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([source[j : j + args.block_size] for j in i])
    y = torch.stack([source[j + 1 : j + args.block_size + 1] for j in i])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, iters=50):
    model.eval()   # switches dropout off
    out = {}
    for name, source in [("train", train_data), ("val", val_data)]:
        losses = [model(*get_batch(source))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def write(model, prompt, length, temperature):
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    return tok.decode(model.generate(idx, length, temperature)[0].tolist())


# ---------------------------------------------------------------------------
# Sampling mode: load a trained model and write text
# ---------------------------------------------------------------------------
if args.sample:
    ckpt = torch.load(run_dir / "model.pt", map_location=device)
    model = GPT(tok.vocab_size, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    missing = [c for c in args.prompt if c not in tok.to_id]
    if missing:
        raise SystemExit(f"These characters aren't in the vocabulary: {missing}")
    print(write(model, args.prompt, args.length, args.temperature))
    raise SystemExit


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
config = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
              block_size=args.block_size, dropout=args.dropout)
if args.n_embd % args.n_head:
    raise SystemExit("n_embd must be divisible by n_head")

model = GPT(tok.vocab_size, **config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
n_params = sum(p.numel() for p in model.parameters())

run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "settings.json").write_text(json.dumps(vars(args), indent=2))
log_file = open(run_dir / "log.csv", "w", newline="")
log = csv.writer(log_file)
log.writerow(["step", "train_loss", "val_loss", "seconds"])
samples_file = open(run_dir / "samples.txt", "w", encoding="utf-8")

print(f"Run '{args.name}' on {device}: {n_params:,} parameters, "
      f"{len(train_data):,} training characters, {args.max_iters} steps")
print(f"{'step':>6}{'train loss':>12}{'val loss':>10}{'time':>8}")

best_val = float("inf")
start = time.time()
for step in range(args.max_iters + 1):
    if step % args.eval_every == 0 or step == args.max_iters:
        losses = estimate_loss(model)
        elapsed = time.time() - start
        print(f"{step:>6}{losses['train']:>12.3f}{losses['val']:>10.3f}{elapsed:>7.0f}s")
        log.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{elapsed:.1f}"])
        log_file.flush()

        model.eval()
        sample = write(model, "\n", 200, 1.0)
        model.train()
        samples_file.write(f"===== step {step} (val loss {losses['val']:.3f}) =====\n{sample}\n\n")
        samples_file.flush()

        # Keep the version with the best val loss: if the model starts
        # overfitting later, we still have its best moment saved.
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "config": config}, run_dir / "model.pt")

    if step == args.max_iters:
        break

    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

log_file.close()
samples_file.close()

print(f"\nBest val loss: {best_val:.3f}  (bigram was about 2.5)")
print(f"Saved to {run_dir}/. Sample from the final model:\n")
model.eval()
print(write(model, "\n", 400, 0.8))
