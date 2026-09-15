"""
Implements iterative text generation from a trained LSTM model:
    1. Take a seed sequence of words.
    2. Encode it with the vocabulary and pad/truncate to `seq_length`.
    3. Predict the next-token probability distribution.
    4. Sample (or take argmax of) the next token.
    5. Append it to the generated sequence and slide the window forward.
    6. Repeat until the desired number of words has been generated.
    7. Decode the generated token ids back into readable text.
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import clean_text, tokenize


def sample_with_temperature(probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample a token index from a probability distribution, adjusted by a
    temperature parameter.

    temperature < 1.0 -> more conservative / repetitive (peakier distribution)
    temperature = 1.0 -> unmodified distribution
    temperature > 1.0 -> more random / creative (flatter distribution)
    """
    probs = np.asarray(probs).astype("float64")
    if temperature <= 0:
        # Degenerate case: temperature 0 means pure greedy (argmax) decoding.
        return int(np.argmax(probs))

    # Apply temperature scaling in log-space for numerical stability.
    log_probs = np.log(probs + 1e-9) / temperature
    scaled = np.exp(log_probs)
    scaled = scaled / np.sum(scaled)
    return int(np.random.choice(len(scaled), p=scaled))


def generate_text(
    model,
    vocab,
    seed_text: str,
    seq_length: int,
    num_words: int = 50,
    temperature: float = 0.8,
) -> str:
    """
    Generate `num_words` new words continuing from `seed_text`.

    Parameters
    ----------
    model : keras.Model
        Trained text-generation model.
    vocab : preprocessing.Vocabulary
        Vocabulary used to encode/decode tokens.
    seed_text : str
        Free-form text used to seed the generator (will be cleaned the same
        way the training corpus was).
    seq_length : int
        The sequence length the model was trained with.
    num_words : int
        How many new words to generate.
    temperature : float
        Sampling temperature (see `sample_with_temperature`).

    Returns
    -------
    str
        The seed text followed by the newly generated words.
    """
    # Clean and tokenize the seed exactly like the training corpus.
    seed_tokens = tokenize(clean_text(seed_text))
    encoded_seed = vocab.encode(seed_tokens)

    generated_ids = list(encoded_seed)
    output_tokens = list(seed_tokens)

    for _ in range(num_words):
        # Keep only the last `seq_length` tokens as the model's context
        # window, left-padding with 0 if the sequence is still short.
        context = generated_ids[-seq_length:]
        padded = pad_sequences([context], maxlen=seq_length, padding="pre")

        probs = model.predict(padded, verbose=0)[0]

        # Index 0 is the reserved <PAD>/<UNK> token (used both for padding
        # and for out-of-vocabulary / rare words collapsed during
        # preprocessing). It is not a real word, so we exclude it from
        # sampling and renormalize the remaining probability mass -- this
        # keeps generated output readable instead of interleaving <PAD>.
        probs = probs.copy()
        probs[0] = 0.0
        probs = probs / probs.sum()

        next_id = sample_with_temperature(probs, temperature=temperature)

        generated_ids.append(next_id)
        output_tokens.append(vocab.idx2word.get(next_id, "<PAD>"))

    return " ".join(output_tokens)


if __name__ == "__main__":
    import argparse
    from tensorflow import keras
    from preprocessing import Vocabulary

    parser = argparse.ArgumentParser(description="Generate text from a trained LSTM model.")
    parser.add_argument("--model", default="models/lstm_text_model.keras")
    parser.add_argument("--vocab", default="models/vocab.pkl")
    parser.add_argument("--seed", required=True, help="Seed text to start generation from.")
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--num_words", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    vocab = Vocabulary.load(args.vocab)

    text = generate_text(
        model,
        vocab,
        args.seed,
        seq_length=args.seq_length,
        num_words=args.num_words,
        temperature=args.temperature,
    )
    print(text)
