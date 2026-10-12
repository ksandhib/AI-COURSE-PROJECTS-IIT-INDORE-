import tensorflow as tf
from data_preparation import create_data_generators
from model import build_model
import os

def train_model(train_dir, validation_dir, epochs=10, batch_size=32):
    """
    Train the model using the provided data directories.

    Parameters:
    - train_dir: Path to training data
    - validation_dir: Path to validation data
    - epochs: Number of training epochs (default: 10)
    - batch_size: Batch size (default: 32)

    Returns:
    - model: Trained model
    - history: Training history
    """

    # Create data generators
    train_generator, validation_generator, _ = create_data_generators(
        train_dir, validation_dir, batch_size=batch_size
    )

    # Build the model
    model = build_model()

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Save the trained model
    model.save('cat_dog_classifier_model.h5')

    return model, history

if __name__ == "__main__":
    train_dir = 'data/train'
    validation_dir = 'data/validation'

    if os.path.exists(train_dir) and os.path.exists(validation_dir):
        model, history = train_model(train_dir, validation_dir)
        print("Model trained and saved successfully.")
    else:
        print("Data directories not found. Please ensure 'data/train' and 'data/validation' exist.")
