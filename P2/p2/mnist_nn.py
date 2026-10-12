import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data Preprocessing
# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model Development
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Evaluation
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Prediction function
def predict_digit(image):
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image, verbose=0)
    return np.argmax(pred)

# Test prediction on a few samples
print("\nPredictions on first 5 test images:")
for i in range(5):
    pred = predict_digit(x_test[i] * 255)  # Denormalize for prediction
    true = np.argmax(y_test[i])
    print(f"Image {i+1}: Predicted {pred}, True {true}")

# Analysis: Confusion Matrix
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)
