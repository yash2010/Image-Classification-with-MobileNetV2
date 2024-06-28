import numpy as np
import os
import arparse
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description='Predict image classes using the trained MobileNetV2 model.')
parser.add_argument('--test_img_path', type=str, required=True, help='Path to the directory containing test images')
args = parser.parse_args()

dataset_dir = "101_ObjectCategories"
test_img_path = args.test_img_path

images = []
labels = []
test_images = []

for subclass in os.listdir(dataset_dir):
    subclass_path = os.path.join(dataset_dir, subclass)
    if os.path.isdir(subclass_path):
        for filename in os.listdir(subclass_path):
            image_path = os.path.join(subclass_path, filename)
            image = imread(image_path)
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image / 255.0
            images.append(image)
            labels.append(subclass)

for filename in os.listdir(test_img_path):
    img_path = os.path.join(test_img_path, filename)
    if img_path.endswith(('png', 'jpg', 'jpeg')):  
        test_image = load_img(img_path, target_size=(224, 224))  
        test_image = img_to_array(test_image)
        test_image = test_image.astype("float32") / 255.0 
        test_images.append(test_image)

test_images = np.array(test_images)

print("Number of images: ", len(test_images))

images = np.array([image for image in images if image.shape == images[0].shape])
labels = np.array(labels)

print("Number of images:", len(images))
print("Number of labels:", len(labels))
if len(images) > len(labels):
    images = images[:len(labels)]
elif len(labels) > len(images):
    labels = labels[:len(images)]
images, labels = shuffle(images, labels, random_state=42)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = tf.keras.utils.to_categorical(labels_encoded)

datagen = ImageDataGenerator(validation_split=0.2)

train_gen = datagen.flow(images, labels_categorical, subset='training', batch_size=32)
val_gen = datagen.flow(images, labels_categorical, subset='validation', batch_size=32)

input_shape = (224, 224, 3)
reg = l2(0.0001)
num_classes = len(label_encoder.classes_)

def create_model(input_shape, reg, num_classes):
    base_model = MobileNetV2(input_shape= input_shape, include_top=False, weights='imagenet')
    for layers in base_model.layers:
        layers.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D()(x)
 
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)
    
    model.summary()
    
    return model

model = create_model(input_shape, reg, num_classes)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['Accuracy'])

history = model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[ModelCheckpoint('model_trans.h5', save_best_only=True), EarlyStopping(patience=3)])

prediction = model.predict(test_images)
predicted_classes = np.argmax(prediction, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)   
print(predicted_labels)




