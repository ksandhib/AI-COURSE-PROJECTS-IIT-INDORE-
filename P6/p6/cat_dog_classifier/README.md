# Cat-Dog Classifier using VGG16 Transfer Learning

This project implements a cat-dog image classifier using transfer learning with the VGG16 model pre-trained on ImageNet.

## Project Structure

```
cat_dog_classifier/
├── requirements.txt          # Python dependencies
├── data_preparation.py       # Data preprocessing and augmentation
├── model.py                  # Model architecture using VGG16
├── train.py                  # Training script
├── evaluate.py               # Evaluation and prediction functions
├── main.py                   # Main script to run the entire pipeline
└── README.md                 # This file
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Create a `data` directory in the project root
   - Inside `data`, create `train`, `validation`, and `test` subdirectories
   - In each of these, create `cats` and `dogs` subdirectories
   - Place your cat and dog images in the respective folders

   Example structure:
   ```
   data/
   ├── train/
   │   ├── cats/
   │   └── dogs/
   ├── validation/
   │   ├── cats/
   │   └── dogs/
   └── test/
       ├── cats/
       └── dogs/
   ```

## Usage

### Training the Model

Run the main script to train the model:
```
python main.py
```

This will:
- Create data generators with augmentation
- Build the VGG16-based model
- Train the model for 10 epochs
- Save the trained model as `cat_dog_classifier_model.h5`

### Evaluating the Model

To evaluate the trained model on test data:
```python
from evaluate import evaluate_model
evaluate_model('cat_dog_classifier_model.h5', 'data/test')
```

### Making Predictions

To predict on a single image:
```python
from evaluate import predict_image
predicted_class, confidence = predict_image('cat_dog_classifier_model.h5', 'path/to/image.jpg')
print(f"Predicted: {predicted_class} with confidence {confidence:.4f}")
```

## Model Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet, top layers removed)
- **Feature Extraction**: Convolutional layers of VGG16 (frozen)
- **Classification Head**:
  - Flatten layer
  - Dense layer (512 units, ReLU activation)
  - Dropout (0.5)
  - Dense layer (256 units, ReLU activation)
  - Dropout (0.5)
  - Output layer (2 units, softmax activation)

## Training Details

- **Optimizer**: SGD with learning rate 0.001 and momentum 0.9
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Data Augmentation**: Rotation, width/height shift, shear, zoom, horizontal flip

## Requirements

- TensorFlow 2.13.0
- NumPy
- Matplotlib
- Pillow

## Notes

- Input images are resized to 150x150 pixels
- The model expects RGB images
- Training may take time depending on your hardware and dataset size
- For better performance, you can experiment with different hyperparameters or add more data augmentation techniques
