"""
utils.py — tokenization, vocabulary, and tensor helpers for the seq2seq model.
"""

import re
import torch

def normalize_contractions(text):
    """
    Expands common informal contractions to a standard format.
    """
    contraction_map = {
        "whats": "what's",
        "im": "i'm",
        "youre": "you're",
        "youve": "you've",
        "youll": "you'll",
        "youd": "you'd",
        "theyre": "they're",
        "theyve": "they've",
        "theyll": "they'll",
        "theyd": "they'd",
        "weve": "we've",
        "well": "we'll",
        "wed": "we'd",
        "ive": "i've",
        "id": "i'd",
        "shouldnt": "shouldn't",
        "shouldve": "should've",
        "couldnt": "couldn't",
        "couldve": "could've",
        "cant": "can't",
        "wont": "won't",
        "dont": "don't",
        "wasnt": "wasn't",
        "werent": "weren't",
        "arent": "aren't",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hadnt": "hadn't",
        "wouldnt": "wouldn't",
        "wouldve": "would've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "isnt": "isn't",
        "its": "it's",
        "thats": "that's",
        "theres": "there's",
        "lets": "let's",
    }
    
    words = text.split()
    normalized = [contraction_map.get(w.lower(), w) for w in words]
    
    return " ".join(normalized)


PAD_TOKEN = "<PAD>"   
SOS_TOKEN = "<SOS>"   
EOS_TOKEN = "<EOS>"  
UNK_TOKEN = "<UNK>"  

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize(sentence: str) -> list[str]:
    """
    Lowercase, keep contractions whole (don't, can't, i'm), split punctuation
    and emojis into individual tokens.
    """
    sentence = sentence.lower().strip()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)*|[^a-z0-9\s]", sentence)


def build_vocab(sentences: list[str]) -> list[str]:
    """
    Build a vocabulary list from a list of sentences in a single pass.
    Returns: SPECIAL_TOKENS + sorted unique words.
    """
    words: set[str] = set()
    for s in sentences:
        words.update(tokenize(s))
    return SPECIAL_TOKENS + sorted(words)


def build_w2i(vocab: list[str]) -> dict[str, int]:
    """Build and return a word→index dict. Call once and reuse."""
    return {w: i for i, w in enumerate(vocab)}


def sentence_to_indices(sentence: str, vocab: list[str],
                         w2i: dict | None = None) -> list[int]:
    """
    Convert a sentence to token indices.
    Pass a pre-built w2i to avoid rebuilding it on every call.
    """
    if w2i is None:
        w2i = build_w2i(vocab)
    return [w2i.get(w, UNK_IDX) for w in tokenize(sentence)]


def indices_to_sentence(indices: list[int], vocab: list[str]) -> str:
    """Convert token indices back to a human-readable string."""
    skip  = {PAD_IDX, SOS_IDX, EOS_IDX}
    words = [vocab[i] for i in indices if i not in skip and i < len(vocab)]
    return " ".join(words)


def encode_with_sos_eos(sentence: str, vocab: list[str],
                         w2i: dict | None = None) -> list[int]:
    """Encode with SOS prepended and EOS appended (decoder target input)."""
    return [SOS_IDX] + sentence_to_indices(sentence, vocab, w2i) + [EOS_IDX]


def pad_sequence(seq: list[int], max_len: int,
                 pad_idx: int = PAD_IDX) -> list[int]:
    """Pad or truncate a sequence to max_len."""
    return seq[:max_len] + [pad_idx] * max(0, max_len - len(seq))


def collate_fn(batch, pad_idx: int = PAD_IDX):
    """
    DataLoader collate: pad src and trg to the longest sequence in the batch.
    batch: list of (src_tensor, trg_tensor)
    """
    src_seqs, trg_seqs = zip(*batch)
    max_src = max(s.size(0) for s in src_seqs)
    max_trg = max(t.size(0) for t in trg_seqs)

    src_padded = torch.zeros(len(batch), max_src, dtype=torch.long)
    trg_padded = torch.zeros(len(batch), max_trg, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, trg_seqs)):
        src_padded[i, : s.size(0)] = s
        trg_padded[i, : t.size(0)] = t

    return src_padded, trg_padded
