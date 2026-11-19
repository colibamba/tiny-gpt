"""
STEP 2: The same bigram model, but LEARNED by a neural network.

In step 1 we counted to get the probabilities. Here the model starts with
random numbers and improves them bit by bit, using the TRAINING LOOP that
every neural network uses, including the biggest AI models:

    repeat many times:
        1. take a random batch of training examples
        2. make predictions                        (forward pass)
        3. measure how wrong they were              (loss)
        4. work out how to change each number to
           make the loss a bit smaller              (backward pass / gradients)
        5. nudge every number in that direction     (optimiser step)

The model is just a 65 x 65 table of numbers: row = current character,
the row's numbers = scores for each possible next character.

Watch the loss fall, and compare where it ends up with step 1's loss.

Run:  py step2_bigram_nn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Tokenizer, load_text, split

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

text = load_text()
tok = Tokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
train, val = split(data)

BATCH_SIZE = 256      # examples per training step
STEPS = 3000
LEARNING_RATE = 0.1   # how big each nudge is


def get_batch(source):
    """Pick random positions; x = a character, y = the one after it."""
    i = torch.randint(len(source) - 1, (BATCH_SIZE,))
    return source[i].to(device), source[i + 1].to(device)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The whole model: a table with one row of scores per character.
        # Starts random. These are the model's "parameters" or "weights".
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.table(x)   # look up the row of scores for each character


model = BigramModel(tok.vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print(f"Parameters (numbers to learn): {sum(p.numel() for p in model.parameters()):,}\n")


@torch.no_grad()
def val_loss():
    """Average loss on validation text the model never trains on."""
    x, y = val[:-1].to(device), val[1:].to(device)
    return F.cross_entropy(model(x), y).item()


print(f"{'step':>6}{'train loss':>12}{'val loss':>10}")
for step in range(STEPS + 1):
    x, y = get_batch(train)                  # 1. batch
    scores = model(x)                        # 2. predict
    loss = F.cross_entropy(scores, y)        # 3. measure (same loss as step 1)
    optimizer.zero_grad()
    loss.backward()                          # 4. gradients
    optimizer.step()                         # 5. nudge

    if step % 300 == 0:
        print(f"{step:>6}{loss.item():>12.3f}{val_loss():>10.3f}")

# Generate some text, exactly like step 1 but using the learned table.
current = torch.tensor([tok.to_id["\n"]], device=device)
out = []
for _ in range(300):
    probs = F.softmax(model(current), dim=-1)   # scores -> probabilities
    current = torch.multinomial(probs, 1)[0]
    out.append(current.item())
print("\n--- Generated text (trained bigram) ---")
print(tok.decode(out))
print("---------------------------------------")
print("\nThe final val loss should be close to step 1's counting loss: training")
print("rediscovered the counts on its own. It can't do better, because it only")
print("ever sees one character. Step 3 fixes that.")
