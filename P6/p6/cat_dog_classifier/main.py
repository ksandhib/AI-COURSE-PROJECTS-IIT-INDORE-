import os
import sys
from data_preparation import create_data_generators
from model import build_model
from train import train_model
from evaluate import evaluate_model, predict_image

def main():
    """
    Main function to run the cat-dog classifier project.
    """

    print("Cat-Dog Classifier using VGG16 Transfer Learning")
    print("=" * 50)

    # Define directories
    train_dir = 'data/train'
    validation_dir = 'data/validation'
    test_dir = 'data/test'
    model_path = 'cat_dog_classifier_model.h5'

    # Check if data directories exist
    if not (os.path.exists(train_dir) and os.path.exists(validation_dir)):
        print("Error: Training and validation data directories not found.")
        print("Please ensure 'data/train' and 'data/validation' directories exist with cat and dog subdirectories.")
        sys.exit(1)

    # Create data generators
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators(train_dir, validation_dir, test_dir)

    # Build the model
    print("Building the model...")
    model = build_model()
    print("Model summary:")
    model.summary()

    # Train the model
    print("Training the model...")
    model, history = train_model(train_dir, validation_dir, epochs=10)

    # Evaluate the model if test data exists
    if test_generator:
        print("Evaluating the model...")
        test_loss, test_accuracy = evaluate_model(model_path, test_dir)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    # Example prediction (you can modify this to predict on your own images)
    print("Example prediction:")
    # Note: Replace 'path/to/your/image.jpg' with an actual image path
    example_image = 'data/test/cats/cat.1.jpg'  # Assuming this path exists
    if os.path.exists(example_image):
        predicted_class, confidence = predict_image(model_path, example_image)
        print(f"Predicted: {predicted_class} with confidence {confidence:.4f}")
    else:
        print("Example image not found. Please provide a valid image path for prediction.")

    print("Project completed successfully!")

if __name__ == "__main__":
    main()
