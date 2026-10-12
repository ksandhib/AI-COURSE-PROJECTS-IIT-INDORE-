import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD

def build_model(input_shape=(150, 150, 3), num_classes=2):
    """
    Build a model using VGG16 for feature extraction and custom fully connected layers for classification.

    Parameters:
    - input_shape: Shape of input images (default: (150, 150, 3))
    - num_classes: Number of classes (default: 2 for cats and dogs)

    Returns:
    - model: Compiled Keras model
    """

    # Load VGG16 without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom fully connected layers
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
