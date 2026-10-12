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

learning_rate = 0.01  # Small learning rate for gradient descent
epochs = 20

errors = []
y_preds = []

for epoch in range(epochs):
    # Feed Forward
    y_pred = x1 * w1 + x2 * w2

    # Error Calculation
    error = (y_actual - y_pred) ** 2
    errors.append(error)
    y_preds.append(y_pred)

    # Back Propagation
    # Gradients
    d_error_d_ypred = -2 * (y_actual - y_pred)
    d_ypred_d_w1 = x1
    d_ypred_d_w2 = x2

    grad_w1 = d_error_d_ypred * d_ypred_d_w1
    grad_w2 = d_error_d_ypred * d_ypred_d_w2

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    print(f"Epoch {epoch+1}: y_pred = {y_pred:.4f}, error = {error:.4f}, w1 = {w1:.4f}, w2 = {w2:.4f}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), errors, marker='o')
plt.title('Error Progression')
plt.xlabel('Epoch')
plt.ylabel('Squared Error')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), y_preds, marker='o')
plt.title('y_pred Progression')
plt.xlabel('Epoch')
plt.ylabel('Predicted y')

plt.tight_layout()
plt.savefig('progression_plots.png')
plt.show()

print(f"\nFinal weights: w1 = {w1:.4f}, w2 = {w2:.4f}")
print(f"Final y_pred = {y_pred:.4f}")
print(f"Actual y = {y_actual}")
print("Plots saved as 'progression_plots.png'")
