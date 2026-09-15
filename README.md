# LSTM Text Generator — Shakespeare

An LSTM-based **word-level text generator** trained on Shakespeare’s complete works. It takes a seed phrase and generates new Shakespeare-style text by predicting the next word iteratively.

### Key Features

* Shakespeare dataset loading and preprocessing
* Word tokenization and vocabulary creation
* LSTM model: **Embedding → LSTM → Dense/Softmax**
* Training with validation split, Early Stopping, and Model Checkpointing
* Temperature-based text generation
* Configurable sequence length, LSTM layers, and hyperparameters
* Bonus comparison of different LSTM architectures

### Project Structure

```text
lstm_text_gen/
├── data/          # Dataset and download script
├── src/           # Preprocessing, model, training & generation
├── models/        # Trained models and vocabulary
├── outputs/       # Generated text samples
├── main.py        # End-to-end pipeline
├── requirements.txt
└── README.md
```

### Dataset

Uses **Tiny Shakespeare**, a ~202K-word public-domain Shakespeare corpus.

### Usage

```bash
pip install -r requirements.txt
python main.py
```

Generate text from an existing model:

```bash
python main.py --skip_training
```

Custom seed:

```bash
python src/generate.py --seed "to be or not to be" --num_words 40 --temperature 0.8
```

### Model Configuration

* Sequence length: 15
* Embedding dimension: 100
* LSTM units: 150
* Dropout: 0.2
* Batch size: 256
* Maximum epochs: 15
* Minimum word frequency: 2

### Results

The model achieved approximately **10.3% validation accuracy** with a best validation loss of **5.78**. It successfully learned Shakespearean vocabulary, character names, and local phrase patterns.

### Limitations

The relatively small dataset and CPU-based training limit long-range coherence and overall text quality. Larger datasets, models, and GPU training could improve the results.
