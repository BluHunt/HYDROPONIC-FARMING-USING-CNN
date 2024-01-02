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


## Plant Disease Detection Model


### Overview
This is animplementation of a Convolutional Neural Network (CNN) for detecting plant diseases. The code uses TensorFlow and Keras to build, train, and evaluate the model. The dataset used for training, validation, and testing consists of images of plants with different health conditions.

### Code Structure

#### 1. **Data Loading and Visualization**: 
- The script begins by loading a set of plant images from a specified directory. It then visualizes a subset of these images using Matplotlib.

#### 2. **Data Preprocessing**: 
- The script uses the `ImageDataGenerator` class from Keras to perform real-time data augmentation. This includes rescaling pixel values and applying various transformations like rotation, shift, shear, and flip to increase the diversity of the training dataset.

#### 3. **Model Architecture**: 
- A simple CNN model is defined using the Sequential API of Keras. It consists of convolutional layers with max-pooling, followed by fully connected layers. The model is compiled with the Adam optimizer and categorical crossentropy loss.

#### 4. **Model Training**: 
- The model is trained using the `fit` method on the training data generator. A ModelCheckpoint callback is used to save the best model during training based on validation accuracy.

#### 5. **Model Evaluation**: 
- The trained model is evaluated on both the training and validation sets to assess its performance.

#### 6. **Test Set Evaluation**: 
- The script assumes the existence of a separate test set directory, and the model is evaluated on this set. Optionally, predictions can be obtained for further analysis.

#### 7. **Loading and Displaying the Model**: 
- The saved model is loaded and its summary is displayed.

#### 8. **Training History Plotting**:
- Matplotlib is used to plot the training and validation accuracy as well as loss over epochs.

#### 9. **Image Prediction Function**: 
- A function is provided to load an image, make predictions using the trained model, and display the image with the predicted label.

### Usage
1. Ensure you have the required libraries installed (`tensorflow`, `PIL`, `matplotlib`).
2. Adjust the paths for the training, validation, and test directories.
3. Run the script to train the model and evaluate its performance.
4. Use the provided image prediction function to make predictions on new images.

Feel free to customize the code based on your dataset and requirements.


## Yield Monitoring Model

Coming Soon...

## Hydroponic Farming Using CNN Website 

Coming Soon...


