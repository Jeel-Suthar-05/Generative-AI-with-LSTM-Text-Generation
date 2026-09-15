"""
Handles all data preparation steps for the LSTM text generator:

    1. Loading the raw text file.
    2. Cleaning it (lowercasing, punctuation removal).
    3. Tokenizing it into words.
    4. Building the vocabulary (word <-> index mappings).
    5. Turning the token stream into fixed-length input/output sequences
       suitable for supervised training of a next-word predictor.

The module is deliberately kept free of any TensorFlow/Keras model code so
that it can be tested and reused independently of the modelling step.
"""

import re
import string
import pickle
from pathlib import Path

import numpy as np


def load_text(file_path: str) -> str:
    """Read a UTF-8 text file from disk and return its contents as a string."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            "See data/download_data.py for instructions on obtaining it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """
    Normalize raw text for word-level modelling:
      - lowercase everything
      - remove punctuation
      - collapse repeated whitespace/newlines into single spaces
    """
    text = text.lower()
    # Remove all punctuation characters (string.punctuation covers
    # ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~ )
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse any run of whitespace (spaces, tabs, newlines) into one space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list:
    """Split cleaned text into a list of word tokens."""
    return text.split(" ")


class Vocabulary:
    """
    Bidirectional word <-> integer-index mapping, built from a token list.

    Index 0 is reserved as a padding / unknown token so that Keras utilities
    (e.g. pad_sequences) behave predictably.
    """

    def __init__(self, tokens: list, min_freq: int = 1):
        from collections import Counter

        counts = Counter(tokens)
        # Keep tokens that occur at least `min_freq` times, sorted by
        # descending frequency (purely cosmetic — helps debugging/inspection)
        vocab_words = [w for w, c in counts.most_common() if c >= min_freq]

        self.word2idx = {"<PAD>": 0}
        for i, word in enumerate(vocab_words, start=1):
            self.word2idx[word] = i
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens: list) -> list:
        """Convert a list of word tokens into a list of integer ids."""
        return [self.word2idx.get(t, 0) for t in tokens]

    def decode(self, ids: list) -> list:
        """Convert a list of integer ids back into word tokens."""
        return [self.idx2word.get(i, "<PAD>") for i in ids]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls.__new__(cls)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        return vocab


def build_sequences(encoded_tokens: list, seq_length: int):
    """
    Build (input, output) pairs for next-word prediction using a sliding
    window of length `seq_length`.

    Given the encoded token stream [t0, t1, t2, t3, t4, ...] and
    seq_length = 3, this produces:
        X = [t0, t1, t2] -> y = t3
        X = [t1, t2, t3] -> y = t4
        ...

    Returns
    -------
    X : np.ndarray of shape (num_sequences, seq_length)
    y : np.ndarray of shape (num_sequences,)
    """
    X, y = [], []
    for i in range(seq_length, len(encoded_tokens)):
        X.append(encoded_tokens[i - seq_length:i])
        y.append(encoded_tokens[i])
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def prepare_dataset(file_path: str, seq_length: int = 10, min_freq: int = 1):
    """
    End-to-end convenience wrapper that runs the full preprocessing
    pipeline described in the task brief:

        raw text -> clean -> tokenize -> vocabulary -> input/output pairs

    Returns
    -------
    X, y : np.ndarray
        Encoded input sequences and next-token targets.
    vocab : Vocabulary
        The fitted vocabulary object (needed later for text generation).
    tokens : list
        The full cleaned/tokenized corpus (useful for inspection/statistics).
    """
    raw_text = load_text(file_path)
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)

    vocab = Vocabulary(tokens, min_freq=min_freq)
    encoded = vocab.encode(tokens)

    X, y = build_sequences(encoded, seq_length)
    return X, y, vocab, tokens


if __name__ == "__main__":
    # Quick manual smoke-test when run directly:
    #   python src/preprocessing.py
    X, y, vocab, tokens = prepare_dataset("data/shakespeare.txt", seq_length=10)
    print(f"Total tokens: {len(tokens):,}")
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Number of training sequences: {len(X):,}")
    print(f"Sample X[0]: {X[0]} -> y[0]: {y[0]}")
    print("Decoded sample:", vocab.decode(list(X[0])), "->", vocab.decode([y[0]]))
