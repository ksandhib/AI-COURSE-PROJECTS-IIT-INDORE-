import os
import zipfile
import urllib.request
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from PIL import Image

# Step 1: Download and extract the dataset
def download_and_extract_dataset():
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    zip_path = "cats_and_dogs_filtered.zip"
    extract_path = "cats_and_dogs_filtered"
    
    if not os.path.exists(extract_path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
    else:
        print("Dataset already exists.")

# Step 2: Organize dataset
def organize_dataset():
    base_dir = "cats_and_dogs_filtered"
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    
    # Create directories if not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    
    return train_dir, validation_dir

# Step 3: Data preprocessing and augmentation
def create_data_generators(train_dir, validation_dir):
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
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

# Step 4: Build the CNN model
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
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# Step 5: Train the model
def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50
    )
    return history

# Step 6: Evaluate and predict
def evaluate_and_predict(model, validation_generator):
    # Evaluate
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    
    # Predict on a sample image (assuming there's a test image)
    # For demo, use a validation image
    sample_image_path = os.path.join("cats_and_dogs_filtered", "validation", "cats", "cat.2000.jpg")
    if os.path.exists(sample_image_path):
        img = Image.open(sample_image_path)
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        if prediction[0] > 0.5:
            print("Prediction: Dog")
        else:
            print("Prediction: Cat")
    else:
        print("Sample image not found for prediction.")

if __name__ == "__main__":
    download_and_extract_dataset()
    train_dir, validation_dir = organize_dataset()
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir)
    model = build_model()
    history = train_model(model, train_generator, validation_generator)
    evaluate_and_predict(model, validation_generator)
