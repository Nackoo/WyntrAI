import numpy as np
import re


def tokenize(sentence: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    return sentence.split()


def stem(word: str) -> str:
    """
    Minimal suffix-stripping stemmer (no external dependencies).
    For a real project, swap this with nltk.stem.PorterStemmer.
    """
    suffixes = ["ing", "tion", "ness", "ment", "ies", "es", "ed", "er", "ly", "s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            return word[: -len(suffix)]
    return word


def bag_of_words(sentence: str, vocabulary: list[str]) -> np.ndarray:
    """
    Convert a sentence into a binary bag-of-words vector.
    Each position is 1 if the stemmed word is in vocabulary, else 0.

    Args:
        sentence:   Raw user input string.
        vocabulary: Sorted list of all known stemmed words.

    Returns:
        np.ndarray of shape (len(vocabulary),), dtype float32.
    """
    tokens = [stem(w) for w in tokenize(sentence)]
    bow = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, vocab_word in enumerate(vocabulary):
        if vocab_word in tokens:
            bow[idx] = 1.0
    return bow