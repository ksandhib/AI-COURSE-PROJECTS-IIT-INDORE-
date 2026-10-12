import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names for visualization
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Visualization: Visualize the distribution of different fashion items
plt.figure(figsize=(10, 6))
plt.hist(train_labels, bins=np.arange(11)-0.5, rwidth=0.8, align='mid')
plt.xticks(range(10), class_names, rotation=45)
plt.xlabel('Fashion Item')
plt.ylabel('Number of Images')
plt.title('Distribution of Fashion Items in Training Dataset')
plt.tight_layout()
plt.savefig('fashion_distribution.png')
plt.show()

# Preprocessing: Normalize the image pixel values (0-255 to 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images for FCNN (28x28 -> 784)
train_images = train_images.reshape((train_images.shape[0], 28*28))
test_images = test_images.reshape((test_images.shape[0], 28*28))

# Convert labels to one-hot encoded format
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Model Architecture: Design a fully connected neural network
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(784,)))  # Input layer + first hidden layer
model.add(layers.Dense(64, activation='relu'))  # Second hidden layer
model.add(layers.Dense(10, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Training: Train the model with validation split
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Performance Evaluation: Assess the model’s performance
train_loss, train_accuracy = model.evaluate(train_images, train_labels)
val_loss, val_accuracy = model.evaluate(test_images, test_labels)  # Using test as validation since no separate val set

print(f"Training Loss: {train_loss}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Visualize the training and validation loss and accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
