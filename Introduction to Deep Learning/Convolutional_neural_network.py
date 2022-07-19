# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:15:23 2022
Programming assignement 3 for Deep learning class
This is a code to perform multi class classification on the CIFAR dataset. 
This is implemented using a convolutional neural network. The weights of the
neural network were optimized using the Adam optimizer.

@author: Glory Programming assignment 3
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# First step, load data
training_data = np.load('training_data.npy')/255
training_label = tf.one_hot(np.load('training_label.npy'), 10).numpy() # one_hot encode labels
training_label = training_label[:,0]                                   # reshape to 50000,10
testing_data = np.load('testing_data.npy')/255
testing_label = tf.one_hot(np.load('testing_label.npy'), 10).numpy()   # one_hot encode labels
testing_label = testing_label[:,0]                                     # reshape to 5000,10

# Define hyperparameters
lr  = 0.0001   # learn rate
batch = 100   # batch size
Epochs =100 # maximum number of epochs

    
# Define classification error per digit
def error_per_digit(y_true, y_pred):
    # Decode predictions and labels
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    # Get classification error per digit using confusion matrix
    confu_matrix = tf.math.confusion_matrix(y_true,y_pred)
    error_per_digit = 1-tf.divide(tf.linalg.tensor_diag_part(confu_matrix), tf.reduce_sum(confu_matrix, 1))
    return error_per_digit

initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)

# Build CNN
model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5), kernel_initializer=initializer, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(32, (5, 5), kernel_initializer=initializer, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), kernel_initializer=initializer, activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(100, (3, 3), kernel_initializer=initializer, activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, kernel_initializer=initializer, use_bias=True, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, kernel_initializer=initializer, use_bias=True, activation='softmax'))
model.summary()


# Define training and testing datasets
train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_label))
test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, testing_label))
# Shuffle and batch datasets
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch)
test_dataset = test_dataset.batch(batch)

# Train model
optimizer = tf.keras.optimizers.Adam(learning_rate= lr) # choose optimizer

# Define loss function and training metrics
model.compile(optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
# Train and store variables
history=model.fit(train_dataset, epochs=Epochs, validation_data=test_dataset)#, callbacks=[callback])

#Get training loss and accuracy
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
# Get testing loss and accuracy
test_accuracy = history.history['val_accuracy']
test_loss = history.history['val_loss']
# Classification error per digit
test_Y_hat= model.predict(test_dataset)
test_class_error = error_per_digit(test_Y_hat, testing_label)

    
# Show final classification error and average
digit = 1
for error in test_class_error:
    print('Classification error for digit', digit, 'is:', error.numpy())
    digit +=1
print('Average classification error is', (np.average(test_class_error)))

# Save model
model.save("trained_model")

# Evaluate saved model
new_model = tf.keras.models.load_model('trained_model')
new_model.evaluate(testing_data, testing_label, batch_size=128)


# Plot training and testing accuracy and loss
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.title('Training and validation accuracy')
plt.show()

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and validation loss')
plt.show()


# Visualize weights
weights= model.layers[0].get_weights()
for i in range(16):
    filte_r = 255*weights[0][:,:,:,i]
    plt.imshow(filte_r)
    plt.grid(None)
    plt.colorbar()
    plt.title('Weights for filter' + str(i))
    plt.show()