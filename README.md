# Hyrdoponic Farming using CNN

## Nutrient Deficiencies Model

This repository contains code for a Convolutional Neural Network (CNN) model designed to identify nutrient deficiencies in hydroponic farming.

### Project Structure

The dataset used for training is stored in the "dataset" folder, which contains four subfolders:
- FN
- -K
- -N
- -P

### Model Overview

The code in this repository consists of the following main components:

#### 1. Visualization of the Dataset
- The code includes visualization of the dataset using Matplotlib to display images from different classes.

#### 2. Model Creation and Training
- Utilizes PyTorch to create a ResNet50 model pretrained on ImageNet and fine-tunes it on the provided dataset.
- The training includes data augmentation techniques such as random rotation to enhance model generalization.

#### 3. Handling Class Imbalance
- Addresses class imbalance in the training data using WeightedRandomSampler to assign appropriate weights to different classes.

#### 4. Evaluation and Testing
- Evaluates the trained model on a validation set and plots the training and validation accuracy over epochs.
- Tests the model on a separate test set and visualizes a subset of predictions against actual labels using Matplotlib.

### Usage

1. **Dependencies:** Ensure PyTorch and other required packages are installed using `pip install torch torchvision torchaudio`.
2. **Dataset:** Place the dataset in the "dataset" folder structured as described above.
3. **Training:** Execute the code to train the model, specifying parameters such as epochs and learning rates.
4. **Evaluation:** Evaluate the model using the provided test set to assess its performance.
5. **Prediction:** Utilize the trained model for predictions on new images.

### File Description

- `main.py`: Contains the main code for data loading, model creation, training, evaluation, and prediction.
- `README.md`: Provides an overview of the project and instructions for usage.
- `lettuce_npk.pth`: Pretrained model file saved after training.

### Notes

- Modify hyperparameters, such as learning rates, batch sizes, or augmentation techniques, for experimentation and potential performance improvements.
- Ensure proper GPU availability for faster model training if using CUDA.

Feel free to contribute, report issues, or suggest improvements.
