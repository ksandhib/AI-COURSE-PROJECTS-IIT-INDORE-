import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define image size
IMG_SIZE = (150, 150)

print("Loading VGG16 model for feature extraction...")
# Load VGG16 model without top layers for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False

print("Adding custom fully connected layers...")
# Add custom fully connected layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)  # 2 classes: cat and dog

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Print model summary
print("\nModel Architecture:")
model.summary()

# Compile the model
print("\nCompiling the model...")
model.compile(
    optimizer=SGD(learning_rate=0.001, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully with SGD optimizer, categorical cross-entropy loss, and accuracy metric.")

# Note: Training requires data in 'data/train' and 'data/validation' directories with cats/ and dogs/ subfolders.
# To train, uncomment the data preprocessing and training sections below.

# Data preprocessing and augmentation (uncomment for training)
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import os
# BATCH_SIZE = 32
# train_dir = 'data/train'
# validation_dir = 'data/validation'
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
# validation_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
# validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
# history = model.fit(train_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE, epochs=10, validation_data=validation_generator, validation_steps=validation_generator.samples // BATCH_SIZE)
# val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
# print(f'Validation Loss: {val_loss}')
# print(f'Validation Accuracy: {val_accuracy}')
# model.save('cat_dog_classifier_model.h5')

# Prediction function
def predict_image(image_path):
    from tensorflow.keras.preprocessing import image
    import numpy as np
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array, verbose=0)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    classes = ['cat', 'dog']
    predicted_class = classes[class_index]

    return predicted_class, confidence

print("\nPrediction function defined. Example usage:")
print("# predicted_class, confidence = predict_image('path/to/test/image.jpg')")
print("# print(f'Predicted: {predicted_class} with confidence {confidence:.2f}')")

print("\nModel setup complete. Ready for training or prediction.")
