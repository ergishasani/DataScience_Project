import os 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import train_test_split 

# Directory containing the dataset
data_dir = '../datasets/' 

# List of brands (class labels)
brands = ['brand1', 'brand2', 'brand3'] 

# Image dimensions and batch size
img_height, img_width = 150, 150 
batch_size = 32 

# Data augmentation and rescaling for training and validation datasets
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

# Creating training data generator
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set to use training subset
) 

# Creating validation data generator
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set to use validation subset
) 

# Loading the pre-trained VGG16 model without the top layers
base_model = VGG16(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet') 

# Freezing the base model to prevent its weights from being updated during training
base_model.trainable = False 

# Defining a Sequential model with the pre-trained base model
model = Sequential([
    # Adding the base VGG16 model
    base_model,
    # Flatten layer to convert the 2D feature maps to a 1D vector
    Flatten(),
    # First dense layer with 128 neurons and ReLU activation
    Dense(128, activation='relu'),
    # Second dense layer with 64 neurons and ReLU activation
    Dense(64, activation='relu'),
    # Output layer with 3 neurons (one for each class) and softmax activation
    Dense(3, activation='softmax')
]) 

# Compiling the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

# Training the model with the training data and validating with validation data
history = model.fit(train_gen, validation_data=val_gen, epochs=10) 

# Saving the trained model
model.save('../models/pretrained_nn_model.h5') 

# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show() 


#Explanation:
#Imports: Necessary libraries are imported, including TensorFlow, Keras, NumPy, Matplotlib, and scikit-learn.

#Directories and Classes: The directory where the dataset is stored and the class labels are defined.

#Image Parameters: Set the height, width of the images, and the batch size for training.##

#Data Augmentation:

#ImageDataGenerator is used to rescale the images and split the data into training and validation sets.
#flow_from_directory creates data generators for the training and validation datasets.
#Base Model:

#The VGG16 model, pre-trained on ImageNet, is loaded without the top (fully connected) layers.
#The weights of the base model are frozen to prevent them from being updated during training.
#Model Definition:

#A Sequential model is defined.
#The pre-trained VGG16 model is added as the base.
#The Flatten layer converts the 2D feature maps to a 1D vector.
#Two Dense layers with ReLU activation are added for learning.
#The output Dense layer uses softmax activation for classification.
#Model Compilation: The model is compiled using the Adam optimizer and categorical cross-entropy loss.

#Model Training: The model is trained using the training data generator and validated using the validation data generator for 10 epochs.

#Model Saving: The trained model is saved to a specified directory.

#Plotting: The training and validation accuracy are plotted over the epochs to visualize the model's performance.