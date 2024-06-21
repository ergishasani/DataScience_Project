import os 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 
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

# Defining a Sequential model with convolutional layers
model = Sequential([
    # First convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    # Max pooling layer with pool size of 2x2
    MaxPooling2D((2, 2)),
    # Flatten layer to convert 2D matrix to 1D vector
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
model.save('../models/conv_dense_nn_model.h5') 

# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show() 
