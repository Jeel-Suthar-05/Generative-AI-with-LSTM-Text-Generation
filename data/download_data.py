"""
Downloads the dataset used for this project: the "Tiny Shakespeare"

Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

Alternative sources:
  - Project Gutenberg: https://www.gutenberg.org/ (search "Shakespeare")
  - Kaggle datasets: https://www.kaggle.com/datasets?search=shakespeare
"""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_PATH = os.path.join(os.path.dirname(__file__), "shakespeare.txt")


def download():
    if os.path.exists(OUT_PATH):
        print(f"Dataset already exists at '{OUT_PATH}'. Skipping download.")
        return
    print(f"Downloading dataset from {URL} ...")
    urllib.request.urlretrieve(URL, OUT_PATH)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Saved to '{OUT_PATH}' ({size_kb:.1f} KB).")


if __name__ == "__main__":
    download()
