import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

def evaluate_model(model_path, test_dir, batch_size=32):
    """
    Evaluate the trained model on test data.

    Parameters:
    - model_path: Path to the saved model
    - test_dir: Path to test data
    - batch_size: Batch size (default: 32)

    Returns:
    - test_loss: Test loss
    - test_accuracy: Test accuracy
    """

    # Load the model
    model = load_model(model_path)

    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    return test_loss, test_accuracy

def predict_image(model_path, image_path):
    """
    Predict the class of a single image.

    Parameters:
    - model_path: Path to the saved model
    - image_path: Path to the image to predict

    Returns:
    - predicted_class: Predicted class ('cat' or 'dog')
    - confidence: Confidence level of the prediction
    """

    # Load the model
    model = load_model(model_path)

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    # Map class index to class name
    class_names = ['cat', 'dog']
    predicted_class = class_names[class_index]

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    return predicted_class, confidence

if __name__ == "__main__":
    model_path = 'cat_dog_classifier_model.h5'
    test_dir = 'data/test'

    if os.path.exists(model_path) and os.path.exists(test_dir):
        evaluate_model(model_path, test_dir)
    else:
        print("Model or test data not found.")

    # Example prediction (assuming an image exists)
    image_path = 'path/to/your/image.jpg'  # Replace with actual path
    if os.path.exists(image_path):
        predict_image(model_path, image_path)
    else:
        print("Image not found for prediction.")
