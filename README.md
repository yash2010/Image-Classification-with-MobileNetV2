# Image Classification with MobileNetV2
This repository contains code for training an image classification model using TensorFlow and MobileNetV2, along with instructions for using the trained model to classify images.

## Dataset
The model is trained on the 101_ObjectCategories dataset, which contains various object categories. Each category has multiple images for training the model.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn
- scikit-image
  
Install dependencies using:

```bash
pip install tensorflow numpy scikit-learn scikit-image
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yash2010/Image-Classification-with-MobileNetV2.git
```
2. Navigate to the project repository

```bash
cd Image-Classification-with-MobileNetV
```

3. Download the 101_ObjectCategories dataset or prepare a similar structured dataset. You can find the link to 101_ObjectCategories dataset [here](https://data.caltech.edu/records/mzrjq-6wc02)
Ensure the dataset is organized into subfolders where each subfolder represents a class/category of images.
Test Images:

Place your test images in the /data/selvaraju/Robotic_project/Lab_pictures directory. Supported formats are .png, .jpg, and .jpeg.
Environment Setup:

Ensure CUDA and cuDNN are installed for GPU support.
Set the GPU visibility using os.environ['CUDA_VISIBLE_DEVICES'] = '1' in your Python script.
Training
Run the training script to train the model:
bash
Copy code
python train.py
This script will train the MobileNetV2 model using the dataset, validating on a subset split from the training data.
Evaluation
After training, the best model will be saved as model_trans.h5.
The model can then be used to predict classes for test images located in /data/selvaraju/Robotic_project/Lab_pictures.
Predictions
Run the prediction script to classify test images:
bash
Copy code
python predict.py
This script will load the trained model and predict the classes for the test images.
Additional Notes
Model Architecture: The model architecture is based on MobileNetV2 with a Global Average Pooling layer and a Dense layer for classification.
Callbacks: The training script uses ModelCheckpoint to save the best model and EarlyStopping to prevent overfitting.
Data Augmentation: ImageDataGenerator is used for data augmentation during training.
Contact
For any issues or questions, please contact Your Name.

Replace <repository_url>, <repository_name>, and update the contact details with your information. This README provides a clear structure for users to understand how to set up the project, train the model, and use it for predictions. Adjust any paths or details as per your specific setup.





