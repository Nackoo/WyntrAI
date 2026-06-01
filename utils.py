"""
utils.py — tokenization, vocabulary, and tensor helpers for the seq2seq model.
"""

import re
import numpy as np
import torch

# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"   # index 0  — padding to equal-length batches
SOS_TOKEN = "<SOS>"   # index 1  — start-of-sequence (fed to decoder first)
EOS_TOKEN = "<EOS>"   # index 2  — end-of-sequence   (stop signal)
UNK_TOKEN = "<UNK>"   # index 3  — unknown word

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize(sentence: str) -> list[str]:
    """Lowercase, strip punctuation (keep apostrophes), split into words."""
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-z0-9\s']", " ", sentence)
    return sentence.split()


def build_vocab(sentences: list[str]) -> list[str]:
    """
    Build a vocabulary list from a list of sentences.
    Returns: sorted list of words, prefixed with special tokens.
    """
    words = set()
    for s in sentences:
        for w in tokenize(s):
            words.add(w)
    return SPECIAL_TOKENS + sorted(words)


def sentence_to_indices(sentence: str, vocab: list[str]) -> list[int]:
    """Convert a sentence to a list of token indices using the vocabulary."""
    w2i = {w: i for i, w in enumerate(vocab)}
    return [w2i.get(w, UNK_IDX) for w in tokenize(sentence)]


def indices_to_sentence(indices: list[int], vocab: list[str]) -> str:
    """Convert a list of token indices back to a human-readable string."""
    skip = {PAD_IDX, SOS_IDX, EOS_IDX}
    words = [vocab[i] for i in indices if i not in skip and i < len(vocab)]
    return " ".join(words)


def encode_with_eos(sentence: str, vocab: list[str]) -> list[int]:
    """Encode + append EOS (used for target sequences during training)."""
    return sentence_to_indices(sentence, vocab) + [EOS_IDX]


def encode_with_sos_eos(sentence: str, vocab: list[str]) -> list[int]:
    """Encode with SOS prepended and EOS appended (decoder target input)."""
    return [SOS_IDX] + sentence_to_indices(sentence, vocab) + [EOS_IDX]


def pad_sequence(seq: list[int], max_len: int, pad_idx: int = PAD_IDX) -> list[int]:
    """Pad or truncate a sequence to max_len."""
    return seq[:max_len] + [pad_idx] * max(0, max_len - len(seq))


def collate_fn(batch, pad_idx: int = PAD_IDX):
    """
    DataLoader collate: pad src and trg sequences to the longest in the batch.
    batch: list of (src_tensor, trg_tensor)
    """
    src_seqs, trg_seqs = zip(*batch)
    src_lens = [s.size(0) for s in src_seqs]
    trg_lens = [t.size(0) for t in trg_seqs]
    max_src  = max(src_lens)
    max_trg  = max(trg_lens)

    src_padded = torch.zeros(len(batch), max_src, dtype=torch.long)
    trg_padded = torch.zeros(len(batch), max_trg, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, trg_seqs)):
        src_padded[i, :s.size(0)] = s
        trg_padded[i, :t.size(0)] = t

    return src_padded, trg_padded


# ── Legacy helpers (kept for any existing callers) ────────────────────────────

def stem(word: str) -> str:
    suffixes = ["ing", "tion", "ness", "ment", "ies", "es", "ed", "er", "ly", "s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)]
    return word


def bag_of_words(sentence: str, vocabulary: list[str]) -> np.ndarray:
    tokens = [stem(w) for w in tokenize(sentence)]
    bow = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, vocab_word in enumerate(vocabulary):
        if vocab_word in tokens:
            bow[idx] = 1.0
    return bow