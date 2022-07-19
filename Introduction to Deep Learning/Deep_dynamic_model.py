# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:26:43 2022
The task for this programming assignment is to estimate the #D joint positions
 of the human body using the body image sequence. The dataset is the a series 
 of short videos recorded by professional actors under 17 different scenarios.
 
 These videos are provided as numpy arrays of dimension 8 x 224 x 224 x 8. Each
 video consists of 8 frames of images of size 224 x 224 x 8. 5,964 videos were
 provided for training and 1,368 videos for testing.
 
 The model used for this task is a deep dynamic model.

@author: glory programming assignment 4
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define mean per joint position error metric
def MPJPE(y_true, y_pred):
    # first find euclidean distance between joint positions
    EN = tf.math.reduce_euclidean_norm((y_true - y_pred), axis = 3)
    # find avarage and convert from m to mm
    mpjpe =  1000*tf.reduce_mean(EN)
    return mpjpe

# First load data
training_data = np.load('data_prog4Spring22/videoframes_clips_train.npy') # 5964 x 16 x 224 x 224 x 3
testing_data = np.load('data_prog4Spring22/videoframes_clips_valid.npy') # 5964 x 16 x 224 x 224 x 3

training_label = np.load('data_prog4Spring22/joint_3d_clips_train.npy') # 1368 x 17 x 3
testing_label = np.load('data_prog4Spring22/joint_3d_clips_valid.npy') # 1368 x 17 x 3

# Define hyperparameters
lr  = 0.001   # learn rate
batch = 2   # batch size
Epochs =10 # maximum number of epochs

# Define training and testing datasets
train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_label))
test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_label))
# Shuffle and batch datasets
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch)
test_dataset = test_dataset.batch(batch)

# Build deep dynamic model
model = models.Sequential()
# first add layers for Resnet(CNN)
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(8, 224, 224, 3)))
#model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.Dropout(0.1))

# # Add layer for LSTM
# model.add(layers.LSTM(128, activation='tanh', return_sequences=True, dropout=0.3))
model.add(layers.Reshape((8,-1)))
model.add(layers.LSTM(512, return_sequences=True))

# Add layers for multilayer perceptron
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.1))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(51, activation='relu'))
#model.add(layers.BatchNormalization())
model.add(layers.Reshape((8,17,3)))
model.summary()

# Train model
optimizer = tf.keras.optimizers.Adam(learning_rate= lr) # choose optimizer

# Define loss function and training metrics
model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=[MPJPE])
# Train and store variables
history=model.fit(train_dataset, epochs=Epochs, validation_data=test_dataset)

#Get training loss and accuracy
train_loss = history.history['loss']
train_accuracy = history.history['MPJPE']
# Get testing loss and accuracy
test_accuracy = history.history['val_MPJPE']
test_loss = history.history['val_loss']

test_Y_hat= model.predict(test_dataset)
test_class_error = MPJPE(test_Y_hat, testing_label)

plt.plot(history.history['MPJPE'], label='training MPJPE')
plt.plot(history.history['val_MPJPE'], label='validation MPJPE')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.title('Training and validation MPJPE')
plt.show()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and validation loss')
plt.show()

