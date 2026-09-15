"""
Defines the LSTM-based text generation architecture and a small factory
function so that different configurations (embedding size, number/size of
LSTM layers, dropout, etc.) can be experimented with easily -- this backs
the "bonus" architecture-comparison requirement in the task brief.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    vocab_size: int,
    seq_length: int,
    embedding_dim: int = 100,
    lstm_units=(150,),
    dropout: float = 0.2,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build and compile an LSTM text-generation model.

    Architecture:
        Input -> Embedding -> [LSTM (+ Dropout)] x N -> Dense(softmax)

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary (size of the softmax output).
    seq_length : int
        Length of the input token sequences.
    embedding_dim : int
        Dimensionality of the word embedding vectors.
    lstm_units : tuple[int]
        Number of units in each stacked LSTM layer. A tuple with more than
        one element stacks multiple LSTM layers (deeper model).
    dropout : float
        Dropout rate applied after each LSTM layer to reduce overfitting.
    learning_rate : float
        Learning rate for the Adam optimizer.

    Returns
    -------
    keras.Model
        A compiled, ready-to-train Keras model.
    """
    model = keras.Sequential(name="lstm_text_generator")
    model.add(layers.Input(shape=(seq_length,)))
    model.add(
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding",
        )
    )

    # Stack one or more LSTM layers. All but the last must return full
    # sequences so they can feed the next LSTM layer.
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        model.add(
            layers.LSTM(
                units,
                return_sequences=return_sequences,
                name=f"lstm_{i + 1}",
            )
        )
        if dropout > 0:
            model.add(layers.Dropout(dropout, name=f"dropout_{i + 1}"))

    # Dense output layer with softmax over the vocabulary -> next-token
    # probability distribution.
    model.add(layers.Dense(vocab_size, activation="softmax", name="output"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick smoke-test: build a small model and print its summary.
    m = build_lstm_model(vocab_size=5000, seq_length=10, lstm_units=(128, 128))
    m.summary()
