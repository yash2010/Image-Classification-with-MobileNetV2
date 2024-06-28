# MobileNetV2 Image Classification
This repository contains the implementation of an image classification model using MobileNetV2. The model is trained on the 101_ObjectCategories dataset and can predict the class of input images.

# Table of Contents
- Introduction
- Dataset
- Installation
- Usage
- Model Architecture
- Training
- Evaluation
- Prediction
- Results
  
## Introduction
This project implements a transfer learning approach using MobileNetV2 for image classification. The model is trained on the 101_ObjectCategories dataset and fine-tuned to recognize different categories of objects.

## Dataset
The dataset used in this project is [101_ObjectCategories](https://data.caltech.edu/records/mzrjq-6wc02). It contains images of objects categorized into 101 different classes.

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yash2010/mobilenetv2-image-classification.git
cd mobilenetv2-image-classification
pip install -r requirements.txt
```
## Usage
1. Prepare the dataset: Download and extract the 101_ObjectCategories dataset into a directory named 101_ObjectCategories.

2. Run the training script: This script will train the model and save the best-performing model to model_trans.h5.

```bash
python train.py
```

3. Run the prediction script: This script will load test images and use the trained model to predict their classes.
```bash
python predict.py --test_img_path path_to_test_images
```

## Model Architecture
The model is based on MobileNetV2, a pre-trained convolutional neural network designed for mobile and edge devices. The final layers are customized for our specific dataset.

- __Base Model:__ MobileNetV2 (pre-trained on ImageNet)

- __Global Average Pooling Layer__

- __Dense Layer:__ Number of units equal to the number of classes, with softmax activation

## Training
The model is trained using the following configuration:

- __Optimizer:__ Adam

- __Learning Rate:__ 0.0001

- __Loss Function:__ Categorical Crossentropy

- __Metrics:__ Accuracy

- __Epochs:__ 20

- __Batch Size:__ 32

- __Callbacks:__ ModelCheckpoint, EarlyStopping

## Evaluation
The model is evaluated on a validation set (20% of the training data) to monitor its performance and to prevent overfitting.

## Prediction
To predict the class of new images, run the predict.py script. It will output the predicted labels for the input test images.
```bash
python predict.py --test_img_path path_to_test_images
```

## Results
The trained model achieves high accuracy on the validation set. Below are sample predictions for the test images:

```less
Predicted Labels: [class1, class2, class3, ...]
```
