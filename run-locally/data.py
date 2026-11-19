"""
Shared helper: downloads the training text and turns it into numbers.

Every step of the project uses this file. The text is "Tiny Shakespeare":
about 1 million characters of Shakespeare's plays (out of copyright).

Models can't read letters, only numbers, so we give each character an ID:
'a' -> 39, 'b' -> 40, and so on. This is called *tokenising*. We use one
token per character to keep things simple. Real models like Llama use
tokens that are whole words or chunks of words.
"""

import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = Path("data") / "input.txt"


def load_text():
    """Return the training text, downloading it the first time."""
    if not DATA_FILE.exists():
        print(f"Downloading Tiny Shakespeare to {DATA_FILE} ...")
        DATA_FILE.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return DATA_FILE.read_text(encoding="utf-8")


class Tokenizer:
    """Converts between text and lists of character IDs."""

    def __init__(self, text):
        self.chars = sorted(set(text))                  # every distinct character
        self.vocab_size = len(self.chars)               # 65 for Shakespeare
        self.to_id = {c: i for i, c in enumerate(self.chars)}
        self.to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.to_id[c] for c in text]

    def decode(self, ids):
        return "".join(self.to_char[i] for i in ids)


def split(data, val_fraction=0.1):
    """
    Split the data: the first 90% for training, the last 10% for validation.

    The model never trains on the validation text. Measuring loss on it tells
    us how well the model does on text it hasn't seen, which is what we
    actually care about. If training loss keeps falling but validation loss
    starts rising, the model is memorising rather than learning: overfitting.
    """
    n = int(len(data) * (1 - val_fraction))
    return data[:n], data[n:]
