# XOR Perceptron Experiment

This project demonstrates the fundamental limitation of single-layer perceptrons and how multi-layer perceptrons (MLPs) can solve non-linearly separable problems using the XOR gate as an example.

## Overview

The XOR (exclusive OR) problem is a classic example in machine learning that highlights the difference between linearly and non-linearly separable problems. This script:

1. **Demonstrates the failure** of a single-layer perceptron on XOR
2. **Implements a solution** using a multi-layer perceptron (MLP)
3. **Visualizes** decision boundaries and training progress

## What is XOR?

The XOR gate outputs `1` when exactly one input is `1`, otherwise it outputs `0`:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

## The Problem: Linearly Separable vs Non-Linearly Separable

- **OR gate** (linearly separable): Can be solved with a single perceptron
  - Class 0: `[0,0]`
  - Class 1: `[0,1]`, `[1,0]`, `[1,1]`
  - A single line can separate these classes

- **XOR gate** (non-linearly separable): Cannot be solved with a single perceptron
  - Class 0: `[0,0]`, `[1,1]`
  - Class 1: `[0,1]`, `[1,0]`
  - No single line can separate these classes

## Implementation

### Single-Layer Perceptron

The `Perceptron` class implements a simple linear classifier:
- **Learning rule**: Perceptron learning algorithm
- **Activation**: Step function (threshold at 0)
- **Limitation**: Can only learn linear decision boundaries

### Multi-Layer Perceptron (MLP)

The `MultiLayerPerceptron` class solves XOR using:
- **Architecture**:
  - Input layer: 2 neurons (for 2 inputs)
  - Hidden layer: 2 neurons with sigmoid activation
  - Output layer: 1 neuron with sigmoid activation
- **Training**: Backpropagation with gradient descent
- **Loss function**: Mean Squared Error (MSE)
- **Activation**: Sigmoid function for non-linear transformations

#### Network Architecture

```
                        Multi-Layer Perceptron for XOR
                        
    Input Layer              Hidden Layer              Output Layer
    ───────────              ─────────────              ────────────
    
        x₁ ──────┐                                       ┌───── y
                 │   ┌──────────┐                      │
                 ├──→│   h₁     │──┐                   │
                 │   │  (σ)     │  │    ┌──────────┐   │
                 │   └──────────┘  ├───→│          │───┘
                 │                  │    │   O₁    │
                 │   ┌──────────┐  │    │   (σ)    │
        x₂ ──────┼──→│   h₂     │──┘    └──────────┘
                 │   │  (σ)     │
                 │   └──────────┘
                 │
                 └──→ (Bias connections not shown)
    
    Legend:
    x₁, x₂  = Input features (2 inputs)
    h₁, h₂  = Hidden neurons (2 neurons with sigmoid activation)
    O₁      = Output neuron (1 neuron with sigmoid activation)
    σ       = Sigmoid activation function
    y       = Final output (XOR result)
    
    Weight Matrices:
    W₁ (2×2): Input → Hidden layer weights
    W₂ (2×1): Hidden → Output layer weights
    b₁, b₂  : Bias terms for each layer
```

**Architecture Details:**
- **Input → Hidden**: Full connections (2 inputs × 2 hidden neurons = 4 weights)
- **Hidden → Output**: Full connections (2 hidden neurons × 1 output = 2 weights)
- **Each connection** has a learnable weight
- **Each layer** has a bias term
- **Total parameters**: 4 + 2 + 2 + 1 = 9 learnable parameters

## Features

1. **Educational Demonstrations**:
   - Shows why single-layer perceptrons fail on XOR
   - Illustrates the need for hidden layers
   - Demonstrates backpropagation learning

2. **Visualizations**:
   - Decision boundary plots (linear vs non-linear)
   - Training error/loss curves
   - Classification accuracy metrics

3. **Detailed Output**:
   - Training progress and convergence status
   - Learned weights and biases
   - Prediction probabilities for each input

## Requirements

```bash
numpy
matplotlib
```

Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

Run the script:
```bash
python xor_perceptron.py
```

The script will:
1. Attempt to train a single-layer perceptron on XOR (will fail)
2. Train a multi-layer perceptron on XOR (will succeed)
3. Display results and visualizations

## Output

### Single-Layer Perceptron Results
- Shows that the perceptron never converges
- Accuracy will be less than 100%
- Decision boundary plot shows a single line (insufficient for XOR)

### Multi-Layer Perceptron Results
- Achieves 100% accuracy on XOR
- Loss decreases to near zero
- Non-linear decision boundary successfully separates the classes
- Detailed predictions with probabilities

## Key Concepts Demonstrated

### 1. Linear Separability
Single-layer perceptrons can only solve problems where classes can be separated by a hyperplane (line in 2D).

### 2. Hidden Layers
Hidden layers transform the input space, making non-linearly separable problems linearly separable in the transformed space.

### 3. Activation Functions
Non-linear activation functions (like sigmoid) are essential for learning complex patterns. Without them, multiple layers would be equivalent to a single layer.

### 4. Backpropagation
The backpropagation algorithm efficiently computes gradients and updates weights through all layers using the chain rule.

## Mathematical Background

### Forward Propagation
1. Hidden layer: `z₁ = X·W₁ + b₁`, `a₁ = σ(z₁)`
2. Output layer: `z₂ = a₁·W₂ + b₂`, `a₂ = σ(z₂)`

Where `σ` is the sigmoid function: `σ(x) = 1 / (1 + e^(-x))`

### Backpropagation
The algorithm computes gradients backward through the network:
1. Compute output error: `dz₂ = a₂ - y`
2. Propagate to hidden layer: `dz₁ = dz₂·W₂ᵀ ⊙ σ'(z₁)`
3. Update weights using gradient descent

## Extensions and Experiments

You can experiment with:
- **Different hidden layer sizes**: Try 3, 4, or more neurons
- **Different learning rates**: See how it affects convergence speed
- **Different activation functions**: Try tanh or ReLU
- **Different architectures**: Add more hidden layers
- **Other non-linearly separable problems**: NAND, XNOR gates

## References

- Perceptron learning algorithm: Rosenblatt (1957)
- XOR problem: Minsky & Papert (1969) - "Perceptrons"
- Backpropagation: Rumelhart, Hinton & Williams (1986)

## License

This is an educational example for learning purposes.
