Sure! Here's a README file for your project:

---

# Pneumonia Detection Using Deep Learning with DenseNet121

This project aims to detect pneumonia from chest X-ray images using a deep learning model based on the DenseNet121 architecture. The dataset is preprocessed with various image augmentation and normalization techniques to enhance performance.

---

## Overview

The project leverages the DenseNet121 model for multi-class classification (Normal, Viral Pneumonia, and Bacterial Pneumonia) of chest X-ray images. The workflow includes data preprocessing, augmentation, model training, and evaluation. This project already trained by many transferlearning models, such as DenseNet, VGG16, EfficienNet, InceptionV3, and ResNet.

---

## Dataset

The dataset includes three classes:
- **Normal**: Healthy chest X-rays.
- **Pneumonia Bacteria**: Chest X-rays with bacterial pneumonia.
- **Pneumonia Virus**: Chest X-rays with viral pneumonia.

The dataset is divided into:
- Training Set
- Validation Set
- Test Set

---

## Preprocessing

### Steps:
1. **Resizing**: Images are resized to 224x224 pixels.
2. **Grayscale Conversion**: Images are converted to grayscale.
3. **Normalization**: Pixel values are scaled to the range [0, 1].
4. **Noise Removal**: Median filters and sharpening filters are applied.
5. **Data Augmentation**: 
   - Rotation, translation, zooming, and flipping.
   - Sobel edge detection to remove unwanted artifacts (e.g., text labels like "R").

### Undersampling and Oversampling:
- **Undersampling**: Reduces the majority class size to match the minority class.
- **Oversampling**: Increases minority class size using data augmentation.

---

## Model Architecture

The DenseNet121 model is used as the base architecture with the following modifications:
- **Base Model**: Pre-trained DenseNet121 (ImageNet weights).
- **Custom Fully Connected Layers**:
  - Flatten Layer.
  - Dense Layer (128 units, ReLU activation).
  - Dropout Layer (0.5 dropout rate).
  - Dense Layer (64 units, ReLU activation).
  - Output Layer (3 units, Softmax activation).

---

## Training

### Parameters:
- **Optimizer**: Adam with a learning rate of 0.0001.
- **Loss Function**: Categorical Cross-Entropy.
- **Metrics**: Accuracy.
- **Early Stopping**: Monitors validation loss with patience of 10 epochs.

### Steps:
1. Preprocessed images are fed into the `ImageDataGenerator` for augmentation.
2. The model is trained on the training set and validated on the test set.
3. Training stops early if no improvement in validation loss is observed.

---

## Evaluation

The model is evaluated on the test set to compute:
- **Accuracy**
- **Loss**
- **Confusion Matrix**
- **Classification Report**

---

## Results

### Performance Metrics:
- **Test Accuracy**: Achieved after 30 epochs (or earlier if early stopping is triggered).
- **Training Time**: ~`<calculated_time>` seconds.

### Model's Strengths:
- Robust to overfitting due to data augmentation.
- High accuracy for both minority and majority classes.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. Preprocess the images:
   ```bash
   python preprocess.py
   ```

3. Train the model:
   ```bash
   python main.py
   ```

4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

---


**Author**: Salsabiela
