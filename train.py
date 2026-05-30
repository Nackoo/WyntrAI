"""
train.py — Build vocabulary from data.json, train ChatNet, save weights.

Usage:
    python train.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import ChatNet
from utils import tokenize, stem, bag_of_words


# ── 1. Load intents ──────────────────────────────────────────────────────────

with open("data.json") as f:
    data = json.load(f)

all_words: list[str] = []
tags: list[str] = []
xy: list[tuple[str, str]] = []   # (pattern, tag)

for intent in data["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((pattern, tag))

# Stem + deduplicate vocabulary
ignore = {"?", "!", ".", ","}
all_words = sorted(set(stem(w) for w in all_words if w not in ignore))
tags = sorted(set(tags))

print(f"Vocabulary size : {len(all_words)}")
print(f"Number of tags  : {len(tags)}")
print(f"Training samples: {len(xy)}")


# ── 2. Build dataset ─────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self):
        self.x = []
        self.y = []
        for pattern, tag in xy:
            bow = bag_of_words(pattern, all_words)
            self.x.append(bow)
            self.y.append(tags.index(tag))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


dataset = IntentDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True)


# ── 3. Initialise model ───────────────────────────────────────────────────────

HIDDEN_SIZE = 64
model = ChatNet(
    input_size=len(all_words),
    hidden_size=HIDDEN_SIZE,
    output_size=len(tags),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ── 4. Training loop ──────────────────────────────────────────────────────────

EPOCHS = 300

print("\nTraining…")
if len(dataset) == 0:
    print("No training samples found in data.json — aborting training.")
    raise SystemExit(1)

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    batch_count = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    if epoch % 50 == 0:
        if batch_count > 0:
            avg = total_loss / batch_count
            print(f"  Epoch {epoch:>4}/{EPOCHS}  loss={avg:.4f}")
        else:
            print(f"  Epoch {epoch:>4}/{EPOCHS}  (no batches)")

print("Done!\n")


# ── 5. Save everything needed for chat.py ────────────────────────────────────

torch.save(
    {
        "model_state": model.state_dict(),
        "input_size": len(all_words),
        "hidden_size": HIDDEN_SIZE,
        "output_size": len(tags),
        "all_words": all_words,
        "tags": tags,
    },
    "model.pth",
)

print("Saved -> model.pth")
print("Run  -> python chat.py")