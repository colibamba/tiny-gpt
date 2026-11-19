"""
STEP 1: The simplest possible language model, built by counting.

A language model does one job: given some text, predict the next character.
This one only looks at the LAST character. It counts, across the whole
training text, how often each character follows each other character.
That's called a "bigram" model.

Then we:
  1. look at what it predicts,
  2. let it write some text,
  3. measure its LOSS, the number that training tries to make smaller.

No AI library needed: this is plain Python.

Run:  py step1_counting.py
"""

import math
import random

from data import Tokenizer, load_text, split

text = load_text()
tok = Tokenizer(text)
data = tok.encode(text)
train, val = split(data)

print(f"Text length: {len(text):,} characters")
print(f"Vocabulary: {tok.vocab_size} different characters: {''.join(tok.chars)!r}\n")

# ---------------------------------------------------------------------------
# 1. Count: counts[a][b] = how many times character b came straight after a
# ---------------------------------------------------------------------------
V = tok.vocab_size
counts = [[1] * V for _ in range(V)]   # start at 1, not 0 (explained below)
for a, b in zip(train, train[1:]):
    counts[a][b] += 1

# Turn counts into probabilities: each row adds up to 1.
probs = [[c / sum(row) for c in row] for row in counts]

# Starting every count at 1 is called "smoothing". Without it, any pair we
# never saw (like "zx") would get probability 0, and the loss below would be
# infinite the moment the validation text contained one.

# ---------------------------------------------------------------------------
# 2. What does it predict?
# ---------------------------------------------------------------------------
print("Most likely next characters:")
for ch in ["q", "t", "h", " ", "\n"]:
    row = probs[tok.to_id[ch]]
    top = sorted(range(V), key=lambda i: row[i], reverse=True)[:3]
    shown = ", ".join(f"{tok.to_char[i]!r} {row[i]:.0%}" for i in top)
    print(f"  after {ch!r}: {shown}")

# ---------------------------------------------------------------------------
# 3. Generate text: repeatedly pick a next character using the probabilities
# ---------------------------------------------------------------------------
random.seed(42)
current = tok.to_id["\n"]
out = []
for _ in range(300):
    current = random.choices(range(V), weights=probs[current])[0]
    out.append(current)
print("\n--- Generated text (counting model) ---")
print(tok.decode(out))
print("---------------------------------------\n")

# ---------------------------------------------------------------------------
# 4. Loss: how surprised is the model by real text it hasn't seen?
# ---------------------------------------------------------------------------
# For each character in the validation text, look up the probability the
# model gave it, and take -log(probability). Confident and right -> small
# number. Thought it unlikely -> big number. Average over all characters.
# This is "cross-entropy loss", the same measure used to train ChatGPT,
# Claude and Llama. Lower is better.
total = sum(-math.log(probs[a][b]) for a, b in zip(val, val[1:]))
loss = total / (len(val) - 1)

# For comparison: a model that guesses every character equally likely.
uniform = math.log(V)

print(f"Loss of random guessing:  {uniform:.3f}")
print(f"Loss of counting model:   {loss:.3f}")
print("\nRemember the counting model's loss. In step 2, a neural network will")
print("learn the same thing by training, and should end up at about the same number.")
