import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import Encoder, Decoder, Seq2Seq
from utils import (
    build_vocab, sentence_to_indices, encode_with_sos_eos,
    collate_fn, PAD_IDX, SOS_IDX, EOS_IDX,
)

with open("data.json") as f:
    data = json.load(f)

pairs: list[tuple[str, str]] = []

for item in data.get("conversations", []):
    src = item.get("input", "").strip()
    replies = item.get("replies", [])
    if isinstance(replies, str):
        replies = [replies]
    elif not replies and "reply" in item:
        replies = [item["reply"]]

    if src:
        for r in replies:
            trg = r.strip()
            if trg:
                pairs.append((src, trg))

if not pairs:
    print("No training pairs found in data.json — aborting.")
    raise SystemExit(1)

print(f"Training pairs : {len(pairs)}")

all_sentences = [p for p, _ in pairs] + [r for _, r in pairs]
vocab = build_vocab(all_sentences)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

class Seq2SeqDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.samples = []
        for src_sent, trg_sent in pairs:
            src = torch.tensor(sentence_to_indices(src_sent, vocab), dtype=torch.long)
            trg = torch.tensor(encode_with_sos_eos(trg_sent, vocab), dtype=torch.long)
            if src.size(0) == 0 or trg.size(0) < 2:
                continue
            self.samples.append((src, trg))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = Seq2SeqDataset(pairs, vocab)
print(f"Usable samples : {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=min(16, len(dataset)),
    shuffle=True,
    collate_fn=collate_fn,
)


EMBED_DIM       = 256 
HIDDEN_SIZE     = 512 
NUM_LAYERS      = 3   
DROPOUT         = 0.1  
EPOCHS          = 150  
LR              = 5e-4
CLIP            = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, dim_feedforward=HIDDEN_SIZE).to(device)
decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, dim_feedforward=HIDDEN_SIZE).to(device)

model = Seq2Seq(
    encoder, decoder,
    sos_idx=SOS_IDX,
    eos_idx=EOS_IDX,
    pad_idx=PAD_IDX,
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

print("\nTraining…")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss   = 0.0
    total_tokens = 0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)
        targets = trg[:, 1:]

        min_len = min(output.size(1), targets.size(1))
        output_flat = output[:, :min_len, :].reshape(-1, vocab_size)
        target_flat = targets[:, :min_len].reshape(-1)

        loss = criterion(output_flat, target_flat)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        non_pad = target_flat.ne(PAD_IDX).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

    avg_loss = total_loss / max(total_tokens, 1)

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={avg_loss:.4f}")

print("Done!\n")


torch.save(
    {
        "encoder_state":   encoder.state_dict(),
        "decoder_state":   decoder.state_dict(),
        "vocab_size":      vocab_size,
        "embed_dim":       EMBED_DIM,
        "hidden_size":     HIDDEN_SIZE,
        "dim_feedforward": HIDDEN_SIZE,
        "num_layers":      NUM_LAYERS,
        "dropout":         DROPOUT,
        "vocab":           vocab,
        "sos_idx":         SOS_IDX,
        "eos_idx":         EOS_IDX,
        "pad_idx":         PAD_IDX,
    },
    "model.pth",
)

print("Saved  -> model.pth")
