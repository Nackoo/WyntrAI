import ast
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import Encoder, Decoder, Seq2Seq
from utils import (
    build_vocab, sentence_to_indices, encode_with_sos_eos, build_w2i,
    collate_fn, PAD_IDX, SOS_IDX, EOS_IDX,
)

# ------------------------------------------------------------------
# 1. AUTO-LOCATING CORNELL CORPUS DATA PATHS
# ------------------------------------------------------------------
def find_cornell_files():
    """Finds the dataset files even if nested in a subfolder."""
    lines_name = "movie_lines.txt"
    conv_name = "movie_conversations.txt"
    
    # Check current directory first
    if os.path.exists(lines_name) and os.path.exists(conv_name):
        return lines_name, conv_name
        
    # Scan subdirectories (like 'cornell movie-dialogs corpus/')
    for root, dirs, files in os.walk('.'):
        if lines_name in files and conv_name in files:
            return os.path.join(root, lines_name), os.path.join(root, conv_name)
            
    raise FileNotFoundError(
        "Could not find 'movie_lines.txt' or 'movie_conversations.txt'. "
        "Please ensure the Cornell dataset is uploaded and unzipped!"
    )

lines_path, conv_path = find_cornell_files()
print(f"🎯 Successfully located dataset inputs:\n -> {lines_path}\n -> {conv_path}\n")

# ------------------------------------------------------------------
# 2. MULTI-TURN CORPUS PARSER
# ------------------------------------------------------------------
def load_cornell_multiturn_pairs(l_path, c_path):
    print("Parsing Cornell Corpus into Multi-Turn sequences...")
    lines = {}
    with open(l_path, "r", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 5:
                lines[parts[0]] = parts[4].strip()

    pairs = []
    with open(c_path, "r", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) == 4:
                line_ids = ast.literal_eval(parts[3].strip())
                
                # Step through sequences chronologically
                for i in range(len(line_ids) - 1):
                    if i == 0:
                        src = lines.get(line_ids[i])
                        trg = lines.get(line_ids[i+1])
                        if src and trg:
                            pairs.append((src, trg, False))
                    elif i >= 2:
                        prev_bot = lines.get(line_ids[i-1])
                        curr_user = lines.get(line_ids[i])
                        trg = lines.get(line_ids[i+1])
                        if prev_bot and curr_user and trg:
                            pairs.append(((curr_user, prev_bot), trg, True))
    return pairs

raw_data = load_cornell_multiturn_pairs(lines_path, conv_path)
print(f"Extracted {len(raw_data):,} sequence segments.")

# ------------------------------------------------------------------
# 3. FILTERING & VOCABULARY GENERATION
# ------------------------------------------------------------------
MAX_SEQ_LEN = 25
filtered_samples = []
all_text_for_vocab = ["<SEP>"] # Force the structural separation token into the vocabulary

for item, trg, has_context in raw_data:
    # Use standard whitespace splitting to safely check lengths without custom tokenizer imports
    trg_toks = trg.split()
    if not (0 < len(trg_toks) <= (MAX_SEQ_LEN - 2)):
        continue
        
    all_text_for_vocab.append(trg)
    
    if has_context:
        curr_user, prev_bot = item
        if 0 < len(curr_user.split()) <= 20 and 0 < len(prev_bot.split()) <= 20:
            filtered_samples.append((item, trg, True))
            all_text_for_vocab.extend([curr_user, prev_bot])
    else:
        if 0 < len(item.split()) <= MAX_SEQ_LEN:
            filtered_samples.append((item, trg, False))
            all_text_for_vocab.append(item)

print(f"Filtered down to {len(filtered_samples):,} multi-turn compatible samples.")

# Rebuild context-aware vocabulary base
vocab = build_vocab(all_text_for_vocab, min_freq=3, max_size=15000)
w2i = build_w2i(vocab)
vocab_size = len(vocab)

# Dynamically locate or assign our structural context boundary token
SEP_IDX = w2i.get("<SEP>", 3) 
print(f"Final Multi-Turn Vocabulary Size: {vocab_size} (SEP Token ID: {SEP_IDX})")

# ------------------------------------------------------------------
# 4. CONTEXT-AWARE DATASET PIPELINE
# ------------------------------------------------------------------
class MultiTurnDataset(Dataset):
    def __init__(self, samples, vocab, w2i, sep_idx):
        self.samples = []
        for item, trg_sent, has_context in samples:
            trg_indices = encode_with_sos_eos(trg_sent, vocab, w2i)
            
            if has_context:
                curr_user, prev_bot = item
                u_idx = sentence_to_indices(curr_user, vocab, w2i)
                b_idx = sentence_to_indices(prev_bot, vocab, w2i)
                # Concatenate: User Prompt + [SEP] + Previous Bot Response
                src_indices = u_idx + [sep_idx] + b_idx
            else:
                src_indices = sentence_to_indices(item, vocab, w2i)
                
            self.samples.append((
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(trg_indices, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = MultiTurnDataset(filtered_samples, vocab, w2i, SEP_IDX)

BATCH_SIZE = 64
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=True)

# ------------------------------------------------------------------
# 5. MODEL ARCHITECTURE SETUP
# ------------------------------------------------------------------
EMBED_DIM       = 256 
HIDDEN_SIZE     = 512 
NUM_LAYERS      = 3   
DROPOUT         = 0.1  
EPOCHS          = 35   
LR              = 5e-4
CLIP            = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Deploying multi-turn training sequence on engine: {device}")

encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, dim_feedforward=HIDDEN_SIZE * 2).to(device)
decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, dim_feedforward=HIDDEN_SIZE * 2).to(device)

model = Seq2Seq(encoder, decoder, sos_idx=SOS_IDX, eos_idx=EOS_IDX, pad_idx=PAD_IDX).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# ------------------------------------------------------------------
# 6. TRAINING LOOP
# ------------------------------------------------------------------
print("\nCommencing Multi-Turn Context Training Loop…")

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
    scheduler.step(avg_loss)
    
    print(f"Epoch {epoch:>2}/{EPOCHS} | Context Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\nMulti-Turn Training Complete!")

# ------------------------------------------------------------------
# 7. SAVE MODEL INTEGRITY
# ------------------------------------------------------------------
torch.save(
    {
        "encoder_state":   encoder.state_dict(),
        "decoder_state":   decoder.state_dict(),
        "vocab_size":      vocab_size,
        "embed_dim":       EMBED_DIM,
        "hidden_size":     HIDDEN_SIZE,
        "dim_feedforward": HIDDEN_SIZE * 2,
        "num_layers":      NUM_LAYERS,
        "dropout":         DROPOUT,
        "vocab":           vocab,
        "sos_idx":         SOS_IDX,
        "eos_idx":         EOS_IDX,
        "pad_idx":         PAD_IDX,
        "w2i":             w2i,
    },
    "model.pth",
)
print("Saved context-aware brain configuration safely to -> model.pth")