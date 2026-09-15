"""
Trains the LSTM text generation model:

    - Splits the input/output pairs into training and validation sets.
    - Trains the model with EarlyStopping (to avoid overfitting) and
      ModelCheckpoint (to persist the best-performing weights).
    - Saves the final model and the fitted vocabulary to disk so that
      `generate.py` can load them later without retraining.
"""

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from preprocessing import prepare_dataset
from model import build_lstm_model


def train_model(
    data_path: str,
    seq_length: int = 10,
    embedding_dim: int = 100,
    lstm_units=(150,),
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 30,
    val_split: float = 0.1,
    min_freq: int = 1,
    model_out: str = "models/lstm_text_model.keras",
    vocab_out: str = "models/vocab.pkl",
    history_out: str = None,
    patience: int = 3,
    max_sequences: int = None,
    seed: int = 42,
):
    """
    Full training pipeline: preprocessing -> train/val split -> model
    building -> training with callbacks -> saving artifacts.

    `max_sequences` can be used to cap the number of training sequences,
    which is useful for quick experiments / limited compute.
    """
    model_dir = os.path.dirname(model_out) or "."
    os.makedirs(model_dir, exist_ok=True)
    if history_out is None:
        history_out = os.path.join(model_dir, "history.json")

    print(f"[1/5] Loading & preprocessing '{data_path}' ...")
    X, y, vocab, tokens = prepare_dataset(data_path, seq_length=seq_length, min_freq=min_freq)
    print(f"      Total tokens: {len(tokens):,} | Vocabulary size: {len(vocab):,}")
    print(f"      Total sequences: {len(X):,}")

    if max_sequences is not None and len(X) > max_sequences:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_sequences, replace=False)
        X, y = X[idx], y[idx]
        print(f"      Subsampled down to {len(X):,} sequences for this run")

    print("[2/5] Splitting into train/validation sets ...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed
    )
    print(f"      Train: {len(X_train):,} | Validation: {len(X_val):,}")

    print("[3/5] Building model ...")
    model = build_lstm_model(
        vocab_size=len(vocab),
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    model.summary()

    print("[4/5] Training ...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_out.replace(".keras", "_best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    print("[5/5] Saving model, vocabulary, and history ...")
    model.save(model_out)
    vocab.save(vocab_out)
    with open(history_out, "w") as f:
        json.dump(history.history, f, indent=2)

    print(f"Done. Model saved to '{model_out}', vocabulary saved to '{vocab_out}'.")
    return model, vocab, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM text-generation model.")
    parser.add_argument("--data", default="data/shakespeare.txt")
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--lstm_units", type=int, nargs="+", default=[150])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--model_out", default="models/lstm_text_model.keras")
    parser.add_argument("--vocab_out", default="models/vocab.pkl")
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        seq_length=args.seq_length,
        embedding_dim=args.embedding_dim,
        lstm_units=tuple(args.lstm_units),
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        min_freq=args.min_freq,
        max_sequences=args.max_sequences,
        model_out=args.model_out,
        vocab_out=args.vocab_out,
    )
