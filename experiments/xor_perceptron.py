"""
Perceptron Attempt for XOR Gate

IMPORTANT: A single-layer perceptron CANNOT solve XOR because XOR is not linearly separable.
This script demonstrates this limitation by attempting to train a perceptron on XOR data.

The XOR problem requires a multi-layer perceptron (MLP) with at least one hidden layer
to solve, as it needs non-linear decision boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt

# Training data: All possible input combinations for a 2-input XOR gate
# Each row represents one training example (input1, input2)
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected outputs for the XOR gate: output is 1 if exactly one input is 1
# [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0
y_xor = np.array([0, 1, 1, 0])


class Perceptron:
    """
    A simple perceptron classifier.
    
    The perceptron is a linear classifier that learns a decision boundary
    by adjusting weights and bias through a training process.
    
    LIMITATION: Cannot solve non-linearly separable problems like XOR.
    """
    def __init__(self, learning_rate=0.1, epochs=100):
        """
        Initialize the perceptron.
        
        Args:
            learning_rate: Step size for weight updates (default: 0.1)
            epochs: Number of training iterations over the dataset (default: 100)
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
        for epoch in range(self.epochs):
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
            
            # Early stopping if we achieve perfect classification
            if errors == 0:
                print(f"Converged after {epoch + 1} epochs!")
                break


class MultiLayerPerceptron:
    """
    Multi-Layer Perceptron (MLP) with one hidden layer.
    
    This neural network can solve non-linearly separable problems like XOR
    by using multiple layers of perceptrons with non-linear activation functions.
    """
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.5, epochs=10000):
        """
        Initialize the MLP.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output neurons
            learning_rate: Step size for weight updates
            epochs: Maximum number of training iterations
        """
        self.lr = learning_rate
        self.epochs = epochs
        
        # Initialize weights with small random values
        # Hidden layer: connects input to hidden layer
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        
        # Output layer: connects hidden layer to output
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip values to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Returns:
            - Hidden layer activations
            - Output layer activations
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Backpropagation to update weights.
        
        Uses gradient descent to minimize the mean squared error.
        """
        m = X.shape[0]  # Number of samples
        
        # Reshape y to match output shape
        y = y.reshape(-1, 1)
        
        # Output layer error
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer error
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def fit(self, X, y):
        """
        Train the MLP using backpropagation.
        """
        for epoch in range(self.epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Calculate loss (mean squared error)
            y_reshaped = y.reshape(-1, 1)
            loss = np.mean((output - y_reshaped) ** 2)
            self.loss_history.append(loss)
            
            # Backpropagation
            self.backward(X, y, output)
            
            # Early stopping if loss is very small
            if loss < 0.001:
                print(f"Converged after {epoch + 1} epochs with loss: {loss:.6f}")
                break
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Returns binary predictions (0 or 1) based on a 0.5 threshold.
        """
        output = self.forward(X)
        return (output >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Return probability predictions (continuous values between 0 and 1).
        """
        return self.forward(X).flatten()


def plot_decision_boundary(X, y, model, title):
    """
    Plot the decision boundary learned by the perceptron.
    
    For XOR, this will show a single line, demonstrating why XOR cannot
    be solved with a single-layer perceptron.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
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


def plot_training_errors(errors_per_epoch, title):
    """
    Plot the number of errors per epoch to visualize training progress.
    
    For XOR, this will show that the perceptron never converges to zero errors.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Number of Errors")
    plt.title(title)
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()


# Attempt to train a perceptron on XOR data
print("=" * 60)
print("Attempting to train a single-layer perceptron on XOR data...")
print("=" * 60)
print("\nNOTE: XOR is NOT linearly separable!")
print("A single-layer perceptron CANNOT solve this problem.\n")

# Create and train a perceptron for XOR
p_xor = Perceptron(learning_rate=0.1, epochs=100)
p_xor.fit(X_xor, y_xor)

# Display training results
print("\nTraining Results:")
print("=" * 60)
print("Weights:", p_xor.weights)
print("Bias:", p_xor.bias)
print("\nPredictions:", p_xor.predict(X_xor))
print("Actual labels:", y_xor)
print("\nAccuracy:", np.mean(p_xor.predict(X_xor) == y_xor) * 100, "%")

# Show that the perceptron never converges
print("\n" + "=" * 60)
print("Training Errors per Epoch:")
print(f"Final epoch errors: {p_xor.errors_per_epoch[-1]}")
print(f"Minimum errors achieved: {min(p_xor.errors_per_epoch)}")
if min(p_xor.errors_per_epoch) > 0:
    print("\n⚠️  The perceptron never achieved zero errors!")
    print("   This confirms that XOR cannot be solved with a single-layer perceptron.")
print("=" * 60)

# Visualize the decision boundary (will show a single line)
print("\nPlotting decision boundary...")
plot_decision_boundary(X_xor, y_xor, p_xor, 
                       "Perceptron Decision Boundary (XOR) - FAILED")

# Visualize training errors
print("Plotting training errors...")
plot_training_errors(p_xor.errors_per_epoch, 
                     "Training Errors per Epoch (XOR - Single Layer Perceptron)")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("To solve XOR, you need a MULTI-LAYER PERCEPTRON (MLP) with:")
print("  - At least one hidden layer")
print("  - Non-linear activation functions (e.g., sigmoid, tanh, ReLU)")
print("  - Backpropagation algorithm for training")
print("=" * 60)


# ============================================================================
# SOLUTION: Multi-Layer Perceptron (MLP) - CAN SOLVE XOR
# ============================================================================
print("\n\n" + "=" * 60)
print("SOLUTION: Training a Multi-Layer Perceptron (MLP) on XOR data...")
print("=" * 60)
print("\nThe MLP uses:")
print("  - Input layer: 2 neurons (for 2 inputs)")
print("  - Hidden layer: 2 neurons with sigmoid activation")
print("  - Output layer: 1 neuron with sigmoid activation")
print("  - Backpropagation for learning\n")

# Create and train MLP for XOR
mlp_xor = MultiLayerPerceptron(input_size=2, hidden_size=2, output_size=1, 
                                learning_rate=0.5, epochs=10000)
mlp_xor.fit(X_xor, y_xor)

# Display training results
print("\nTraining Results (MLP):")
print("=" * 60)
print(f"Final loss: {mlp_xor.loss_history[-1]:.6f}")
print("\nWeights (Hidden Layer):")
print(mlp_xor.W1)
print("\nBiases (Hidden Layer):")
print(mlp_xor.b1)
print("\nWeights (Output Layer):")
print(mlp_xor.W2)
print("\nBias (Output Layer):")
print(mlp_xor.b2)

# Make predictions
predictions = mlp_xor.predict(X_xor)
probabilities = mlp_xor.predict_proba(X_xor)

print("\n" + "-" * 60)
print("Predictions:")
print("-" * 60)
for i, (xi, true_label, pred, prob) in enumerate(zip(X_xor, y_xor, predictions, probabilities)):
    status = "✓" if pred == true_label else "✗"
    print(f"{status} Input: {xi} -> True: {true_label}, Predicted: {pred} (prob: {prob:.4f})")

print(f"\nAccuracy: {np.mean(predictions == y_xor) * 100:.1f}%")
print("=" * 60)

# Visualize the decision boundary (will show non-linear separation)
print("\nPlotting decision boundary for MLP...")
plot_decision_boundary(X_xor, y_xor, mlp_xor, 
                       "MLP Decision Boundary (XOR) - SUCCESS!")

# Visualize training loss
print("Plotting training loss...")
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mlp_xor.loss_history) + 1), mlp_xor.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss per Epoch (MLP - XOR)")
plt.yscale('log')  # Log scale for better visualization
plt.grid(True)
plt.show()

print("\n" + "=" * 60)
print("SUCCESS! The Multi-Layer Perceptron CAN solve XOR!")
print("=" * 60)
print("\nKey differences from single-layer perceptron:")
print("  ✓ Multiple layers allow learning non-linear patterns")
print("  ✓ Hidden layer creates feature transformations")
print("  ✓ Non-linear activation functions enable complex decision boundaries")
print("=" * 60)
