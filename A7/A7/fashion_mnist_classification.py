import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os

# Load Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for visualization
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize different samples from the dataset
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig('fashion_mnist_samples.png')  # Save visualization
plt.close()

# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert the output labels to categorical format
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Split test into validation and test
val_images = test_images[:5000]
val_labels = test_labels[:5000]
test_images = test_images[5000:]
test_labels = test_labels[5000:]

# Model Building
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard callback
log_dir = "logs/fit/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels), callbacks=[tensorboard_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Save the trained model
model.save('fashion_mnist_model.h5')

# Load the saved model and verify
loaded_model = keras.models.load_model('fashion_mnist_model.h5')

# Check architecture
print("Original model summary:")
model.summary()
print("\nLoaded model summary:")
loaded_model.summary()

# Check parameters
original_params = model.count_params()
loaded_params = loaded_model.count_params()
print(f"\nOriginal model parameters: {original_params}")
print(f"Loaded model parameters: {loaded_params}")
print(f"Parameters match: {original_params == loaded_params}")

# Verify predictions
predictions = loaded_model.predict(test_images[:5])
print("\nPredictions for first 5 test images:")
for i in range(5):
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(test_labels[i])
    print(f"Predicted: {class_names[predicted_label]}, True: {class_names[true_label]}")
