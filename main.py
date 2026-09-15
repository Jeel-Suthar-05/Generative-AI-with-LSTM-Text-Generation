"""
Running this script will:
    1. Preprocess the dataset (clean, tokenize, build input/output pairs).
    2. Build and train the LSTM model (with early stopping + checkpointing).
    3. Generate sample text from several different seed inputs.
    4. Write the generated samples to outputs/sample_generated_text.txt.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import Vocabulary
from model import build_lstm_model
from train import train_model
from generate import generate_text
from tensorflow import keras


DEFAULT_SEEDS = [
    "to be or not to be",
    "romeo where art thou",
    "the king said unto",
    "shall i compare thee to",
]


def main():
    parser = argparse.ArgumentParser(description="LSTM Text Generation - full pipeline")
    parser.add_argument("--data", default="data/shakespeare.txt")
    parser.add_argument("--seq_length", type=int, default=15)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--lstm_units", type=int, nargs="+", default=[150])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--num_words", type=int, default=40, help="Words to generate per seed")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--model_out", default="models/lstm_text_model.keras")
    parser.add_argument("--vocab_out", default="models/vocab.pkl")
    parser.add_argument("--output_file", default="outputs/sample_generated_text.txt")
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and load an already-trained model + vocabulary.",
    )
    args = parser.parse_args()

    if args.skip_training:
        print(f"Loading existing model from '{args.model_out}' ...")
        model = keras.models.load_model(args.model_out)
        vocab = Vocabulary.load(args.vocab_out)
    else:
        model, vocab, _ = train_model(
            data_path=args.data,
            seq_length=args.seq_length,
            embedding_dim=args.embedding_dim,
            lstm_units=tuple(args.lstm_units),
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            min_freq=args.min_freq,
            model_out=args.model_out,
            vocab_out=args.vocab_out,
        )

    print("\n" + "=" * 70)
    print("GENERATING SAMPLE TEXT")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        for seed in DEFAULT_SEEDS:
            generated = generate_text(
                model,
                vocab,
                seed,
                seq_length=args.seq_length,
                num_words=args.num_words,
                temperature=args.temperature,
            )
            block = f"Seed: '{seed}'\nGenerated: {generated}\n"
            print(block)
            f.write(block + "\n")

    print(f"Sample outputs written to '{args.output_file}'")


if __name__ == "__main__":
    main()
