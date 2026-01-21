"""
Perceptron Implementation for OR Gate

This script implements a simple perceptron neural network to learn the OR gate logic.
The perceptron is trained on all possible inputs for a 2-input OR gate.
"""
import numpy as np
import matplotlib.pyplot as plt

# Training data: All possible input combinations for a 2-input OR gate
# Each row represents one training example (input1, input2)
X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected outputs for the OR gate: output is 1 if at least one input is 1
# [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 1
y_or = np.array([0, 1, 1, 1])

class Perceptron:
    """
    A simple perceptron classifier.
    
    The perceptron is a linear classifier that learns a decision boundary
    by adjusting weights and bias through a training process.
    """
    def __init__(self, learning_rate=0.1, epochs=20):
        """
        Initialize the perceptron.
        
        Args:
            learning_rate: Step size for weight updates (default: 0.1)
            epochs: Number of training iterations over the dataset (default: 20)
        """
        self.lr = learning_rate  # Learning rate for weight updates
        self.epochs = epochs  # Number of training epochs
        self.weights = None  # Will be initialized during training
        self.bias = None  # Will be initialized during training
        self.errors_per_epoch = []  # Track number of errors per epoch
    def predict(self, X):
        """
        Make predictions for input data.
        
        Args:
            X: Input features (can be a single sample or array of samples)
            
        Returns:
            Binary predictions (0 or 1) based on the linear decision boundary
        """
        # Compute linear combination: w1*x1 + w2*x2 + ... + bias
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply step function: output 1 if >= 0, else 0
        return np.where(linear_output >= 0, 1, 0)
    def fit(self, X, y):
        """
        Train the perceptron on the given data.
        
        Args:
            X: Training input features
            y: Training target labels (binary: 0 or 1)
        """
        n_samples, n_features = X.shape
        # Initialize weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Train for specified number of epochs
        for _ in range(self.epochs):
            errors = 0  # Count misclassifications in this epoch
            # Process each training sample
            for xi, target in zip(X, y):
                # Compute prediction for current sample
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                
                # Calculate update: (target - prediction) * learning_rate
                # Only updates when prediction is wrong
                update = self.lr * (target - y_pred)
                
                # Update weights and bias using perceptron learning rule
                self.weights += update * xi
                self.bias += update
                
                # Count this as an error if update was non-zero
                errors += int(update != 0)
            
            # Record number of errors for this epoch
            self.errors_per_epoch.append(errors)

# Create and train a perceptron for the OR gate
p_or = Perceptron(learning_rate=0.1, epochs=20)
p_or.fit(X_or, y_or)

# Display training results
print("Weights:", p_or.weights)
print("Bias:", p_or.bias)
print("Predictions:", p_or.predict(X_or))

def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

    for label in np.unique(y):
        pts = X[y == label]
        plt.scatter(pts[:, 0], pts[:, 1],
                    s=100, edgecolor='black',
                    label=f"Class {label}")

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_decision_boundary(X_or, y_or, p_or, "Perceptron Decision Boundary (OR)")