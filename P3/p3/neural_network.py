import numpy as np
import matplotlib.pyplot as plt
import random

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Sample Data
x1 = 3
x2 = 2
y_actual = 17

# Initialize random weights between 1 and 10
w1 = random.uniform(1, 10)
w2 = random.uniform(1, 10)

learning_rate = 0.01
epochs = 20

errors = []
ypreds = []

for epoch in range(epochs):
    # Feed Forward
    y_pred = x1 * w1 + x2 * w2

    # Error Calculation
    error = (y_actual - y_pred) ** 2

    # Back Propagation
    grad_w1 = -2 * (y_actual - y_pred) * x1
    grad_w2 = -2 * (y_actual - y_pred) * x2

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    # Store for plotting
    errors.append(error)
    ypreds.append(y_pred)

    print(f"Epoch {epoch+1}: y_pred = {y_pred:.4f}, error = {error:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), errors, marker='o')
plt.title('Error Progression')
plt.xlabel('Epoch')
plt.ylabel('Squared Error')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), ypreds, marker='o', label='y_pred')
plt.axhline(y=y_actual, color='r', linestyle='--', label='y_actual')
plt.title('y_pred Progression')
plt.xlabel('Epoch')
plt.ylabel('y_pred')
plt.legend()

plt.tight_layout()
plt.savefig('neural_network_plots.png')
plt.show()
