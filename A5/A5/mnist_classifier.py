import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Preprocessing
# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoded format
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model Development
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Input layer: flatten 28x28 images
    Dense(128, activation='relu'),   # Hidden layer with 128 units and ReLU
    Dense(64, activation='relu'),    # Another hidden layer
    Dense(10, activation='softmax')  # Output layer with 10 units and softmax
])

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# Training
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Prediction function
def predict_digit(image):
    """
    Predict the digit class for a single 28x28 image.
    Returns the predicted class and probabilities.
    """
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions[0]

# Example prediction on a test image
sample_image = x_test[0] * 255.0  # Denormalize for display
predicted_class, probabilities = predict_digit(sample_image)
print(f"Predicted Digit: {predicted_class}")
print(f"Probabilities: {probabilities}")

# Analysis
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")

# Check for class imbalance (simple check)
unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
print("Class distribution in training set:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples")
