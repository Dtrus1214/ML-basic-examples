# Digit detector (simple) — feedforward + raw PNGs

Minimal digit classifier **0–9** using a **feedforward (fully connected)** network and a **raw image dataset** of PNG files (e.g. ~100 images).

## Dataset: raw PNGs

Put your images in folder-per-class layout:

```
data/
  train/  0/  *.png   (digit 0)
          1/  *.png   (digit 1)
          ...
          9/  *.png   (digit 9)
  val/    0/  *.png
          ...
          9/  *.png
  test/   0/  *.png
          ...
          9/  *.png
```

Any size/format is fine; they are resized to 28×28 and converted to grayscale. With about **100 PNGs** total (e.g. 70 train, 15 val, 15 test), you can train the small feedforward model.

## Quick start (sample data from MNIST)

Create ~90 PNGs from MNIST (3 per digit per split) and train:

```bash
pip install -r requirements.txt
python create_sample_data.py
python train.py
```

Options for `create_sample_data.py`:

- `--root data` — where to create `data/train`, `data/val`, `data/test`
- `--per-class 4` — 4 images per digit per split (120 PNGs total)

## Train on your own PNGs

1. Create `data/train/0` … `data/train/9`, `data/val/0` … `data/val/9`, `data/test/0` … `data/test/9`.
2. Add your digit PNGs into the right folder (e.g. images of "3" in `data/train/3/`).
3. Run:

```bash
python train.py
```

Options: `--epochs`, `--batch-size`, `--lr`, `--data-dir`, `--save model_best.pt`, `--hidden 128`.

## Predict

```bash
python predict.py --checkpoint model_best.pt --image path/to/digit.png
```

## Training model structure

The model is a **feedforward network**: each layer is made of units that work like a single neuron (perceptron):

- **Inputs** (e.g. pixel values) are multiplied by **weights** and summed with a **bias**.
- The result is passed through an **activation** function to produce an **output**.

Single neuron (perceptron) structure:

| Inputs (x₁, x₂, … xₙ) | → | Weights (w₁…wₙ) | → | Σ + bias *b* | → | Activation *g*(·) | → | Output ŷ |
|-----------------------|---|----------------|---|--------------|---|-------------------|---|---------|

**ŷ = g(w · x + b)**

Our digit model stacks three such layers (flattened image → hidden → hidden → 10 classes):

```
                    Weights w₁…w₇₈₄
  Image (28×28)  ──────────────────►  Σ + b  ──► ReLU  ──►  (×128 units)
  flattened to 784                                 │
                                                   ▼
                    Weights                        Σ + b  ──► ReLU  ──►  (×128 units)
                                                   │
                                                   ▼
                    Weights                        Σ + b  ──►  (×10 logits)  ──► digit 0–9
```

So: **784 inputs → 128 hidden (ReLU) → 128 hidden (ReLU) → 10 outputs**. Training updates the weights and biases to minimize classification error.

## Model (summary)

- **Feedforward only**: no convolutions. Image is flattened to 784 values (28×28), then:
  - Linear(784 → 128) → ReLU → Dropout
  - Linear(128 → 128) → ReLU → Dropout
  - Linear(128 → 10)
- Input: 28×28 grayscale (resized from your PNGs).
- Output: class 0–9.

Files: `model.py`, `dataset.py`, `train.py`, `predict.py`, `create_sample_data.py`.
