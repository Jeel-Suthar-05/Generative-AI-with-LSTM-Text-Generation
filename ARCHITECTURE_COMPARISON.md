# Bonus: Architecture Comparison

The task brief asks us to experiment with different model architectures /
sequence lengths and report how they affect generated text quality. Two
configurations were trained on the same dataset (Shakespeare's complete
works, ~202K words, vocabulary trimmed to words occurring ≥2 times → 6,755
tokens) so that the results below are comparable.

## Configurations

| | Model A (primary) | Model B (bonus) |
|---|---|---|
| Sequence length | 15 | 8 |
| Embedding dim | 100 | 100 |
| LSTM layers | 1 × 150 units | 2 × 128 units (stacked) |
| Dropout | 0.2 | 0.3 |
| Trainable params | 1,846,105 | 1,795,727 |
| Epochs run | 12 (early-stopped, best = epoch 9) | 6 (fixed) |
| Final train loss | 5.13 | 5.86 |
| Final val loss | 5.78 (best: 5.783 @ epoch 9) | 6.00 |
| Final train accuracy | 12.1% | 8.3% |
| Final val accuracy | 10.7% | 8.3% |

(Both are top-1 next-word accuracy over a 6,755-word vocabulary — for
context, random guessing would score ≈0.015%, so both models have learned
substantial structure.)

## Sample outputs, same seed, both models (temperature = 0.8)

**Seed: "to be or not to be"**

- **Model A** (1×150, seq_len=15): *"to be or not to be thy country and be a and you were made at some own bed fear my lord why buckingham what i cannot tear the place and settled their course disdains and"*
- **Model B** (2×128, seq_len=8): *"to be or not to be his husband and the arms to sue of death to be a fault and my keeper rivers but not they that deny forth to we command the sovereigns key it"*

**Seed: "romeo where art thou"**

- **Model A**: *"romeo where art thou my head and thy lowly windows which done is the earth of answer for you point his opposers subject when she is possessd into my care to do it die"*
- **Model B**: *"romeo where art thou young arms should them my friends aedile fair villain but no is a talk of which aedile where who and him it is so with my land what you will"*

## Observations

1. **Longer context window (seq_length) helped more than a deeper stack, given the same training budget.** Model A, with a single LSTM layer but a longer 15-word context window, reached a lower validation loss (5.78 vs 6.00) and higher accuracy than the deeper 2-layer model B, even though B ran fewer epochs. A longer window gives the model more grammatical and thematic context per prediction, and on a modest-sized corpus (~200K words) that mattered more than architectural depth.
2. **The 2-layer stack is harder to train in the same number of epochs.** Stacked LSTMs have more representational capacity but also a harder optimization landscape (vanishing/exploding gradients across two recurrent layers), so they need more epochs (or a lower learning rate / gradient clipping) to catch up. With only 6 epochs, Model B was still improving steadily at the end (loss still dropping ~0.11/epoch) and had not converged.
3. **Both models learn "Shakespearean" surface form quickly** (archaic pronouns, verse-like phrase rhythm, character names such as "buckingham", "romeo", "aedile") long before they learn long-range coherence — this is typical for word-level LSTM language models trained on relatively little data. Neither model produces fully grammatical, semantically coherent sentences at this scale; that would need either much more training data/time, a larger hidden size, or a smaller vocabulary (e.g. character-level modelling, or capping vocabulary more aggressively).
4. **Temperature has a clear, consistent effect across both architectures**: temperature 0.5 produces safer, more repetitive phrases ("the king of ... the world of ..."), while temperature 1.0 produces more surprising, less grammatical word choices with more rare proper nouns. Temperature 0.8 was the best trade-off for readability in both models.

## What we'd try next with more time/compute

- Train Model A for more epochs with a learning-rate schedule / gradient clipping to see if the deeper architecture eventually overtakes it.
- Try a larger hidden size (e.g. 256) with regularization (dropout, recurrent_dropout) to fight overfitting.
- Try character-level modelling, which typically produces more fluent local grammar (word spelling is always valid) at the cost of longer training and less coherent higher-level structure.
- Use beam search or nucleus (top-p) sampling instead of plain temperature sampling for more coherent generation.
