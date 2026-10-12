import os
import zipfile
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Step 1: Download and extract the dataset
def download_and_extract_dataset():
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_path = "cats_and_dogs_filtered.zip"
    extract_path = "cats_and_dogs_filtered"

    if not os.path.exists(extract_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
    else:
        print("Dataset already exists.")

    return extract_path

# Step 2: Organize dataset and set up generators
def setup_data_generators(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Normalization for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator

# Step 3: Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model

# Step 4: Train the model
def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # Adjust based on dataset size
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )
    return history

# Step 5: Evaluate and plot
def evaluate_model(model, validation_generator, history):
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Plot training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# Step 6: Prediction function
def predict_image(model, image_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)
    class_names = ['cat', 'dog']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

# Main execution
if __name__ == "__main__":
    base_dir = download_and_extract_dataset()
    train_generator, validation_generator = setup_data_generators(base_dir)
    model = build_model()
    history = train_model(model, train_generator, validation_generator)
    evaluate_model(model, validation_generator, history)

    # Save the model for future predictions
    model.save('cats_and_dogs_model.h5')
    print("Model saved as cats_and_dogs_model.h5")

    # Example prediction (assuming you have an image)
    # predicted_class, confidence = predict_image(model, 'path_to_image.jpg')
    # print(f"Predicted: {predicted_class} with confidence {confidence}")
