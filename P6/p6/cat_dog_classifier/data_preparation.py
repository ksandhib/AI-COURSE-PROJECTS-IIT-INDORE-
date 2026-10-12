import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_data_generators(train_dir, validation_dir, test_dir=None, img_size=(150, 150), batch_size=32):
    """
    Create data generators for training, validation, and optionally test sets with augmentation.

    Parameters:
    - train_dir: Path to training data directory
    - validation_dir: Path to validation data directory
    - test_dir: Path to test data directory (optional)
    - img_size: Tuple for image size (default: (150, 150))
    - batch_size: Batch size for generators (default: 32)

    Returns:
    - train_generator: Training data generator
    - validation_generator: Validation data generator
    - test_generator: Test data generator (if test_dir provided)
    """

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

    # Only rescaling for validation and test
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255) if test_dir else None

    # Flow from directory for training
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Flow from directory for validation
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = None
    if test_dir:
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

    return train_generator, validation_generator, test_generator

if __name__ == "__main__":
    # Example usage (assuming directories exist)
    train_dir = 'data/train'
    validation_dir = 'data/validation'
    test_dir = 'data/test'

    if os.path.exists(train_dir) and os.path.exists(validation_dir):
        train_gen, val_gen, test_gen = create_data_generators(train_dir, validation_dir, test_dir)
        print("Data generators created successfully.")
    else:
        print("Data directories not found. Please ensure 'data/train', 'data/validation', and 'data/test' exist.")
