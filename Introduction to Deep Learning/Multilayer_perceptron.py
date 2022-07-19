# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:06:13 2022
Programming assignement 2 for Deep learning class
This is a code to perform multi class classification on a set of handwritten images
of 5 numbers using a multi-layer perceptron and back propagation. The neural network used here has an input layer, two hidden layers with ReLU
activation and an output layer with softmax activation function. The neural network is
optimized using Stochastic Gradient Descent.

@author: glory
"""
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random as r
import tensorflow as tf

# Hyperparameters
batch_size = 50   # batch size
learn_rate = 1e-3 # learn rate
epochs = 13000      # maximum number of epochs

# Initialize weights using random numbers
W_1 = 0.1*np.random.rand(784,100)
W_10 = 0.1*np.random.rand(100,1)
W_2 = 0.1*np.random.rand(100,100)
W_20 = 0.1*np.random.rand(100,1)
W_3 = 0.1*np.random.rand(100,10)
W_30 = 0.1*np.random.rand(10,1)
theta = [W_1, W_10, W_2, W_20, W_3, W_30]

# read images
def load_images(path_list):
    number_samples = len(path_list)
    Images = []
    for each_path in path_list:
        img = plt.imread(each_path)
        # divided by 255.0
        img = img.reshape(784, 1) / 255.0
        Images.append(img)
    data = tf.convert_to_tensor(np.array(Images).reshape(number_samples, 784), dtype=tf.float32)
    return data

# The cross entropy loss function
def loss(Y_hat, Y):
    Y = Y.numpy()
    L_NLL = tf.reduce_mean(tf.reduce_sum(-tf.math.multiply(Y, tf.math.log(Y_hat)), axis=1))
    return L_NLL

# Forward propagation
def forward_prop(X, theta):
    X = X.numpy()
    N = X.shape[0]
    # Input layer to first hidden layer
    Z_1 = theta[0].T @ X.T + theta[1] @ np.ones((1,N))   # 100x50
    # Relu activation
    H_1 = tf.nn.relu(Z_1)   
    # Second hidden layer
    Z_2 = theta[2].T @ H_1.numpy() + theta[3] @ np.ones((1,N)) # 100x50
    # ReLu activation
    H_2 = tf.nn.relu(Z_2)
    # Output layer
    Z_3 = theta[4].T @ H_2.numpy() + theta[5] @ np.ones((1,N))  # 10x50
    # Softmax
    Y_hat = 1e-20+ np.exp(Z_3)/np.sum(np.exp(Z_3), axis=0) # 10x50
    return Y_hat.T, H_1.numpy(), H_2.numpy()

# Gradients of all the needed weights and outputs with L1 regularization
def gradients(X, Y_hat, Y, theta, H_2, H_1):
    X= X.numpy()
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    grad_Y_hat = Y_hat - Y    # 50x10
    grad_W3 = H_2 @ grad_Y_hat # 100x10
    grad_W30 = np.ones((1,50)) @ grad_Y_hat  # 10x1
    grad_H2 = theta[4] @ grad_Y_hat.T          # 100x50
    grad_W2 = H_1 @ grad_H2.T                  # 100x100
    grad_W20= grad_H2 @ np.ones((50,1))       # 100x1
    grad_H1 = theta[2] @ grad_H2             # 100x50
    grad_W1 = X.T @ grad_H1.T                    # 784x100
    grad_W10 = grad_H1 @ np.ones((50,1))      # 100x1
    grads = [grad_W1, grad_W10, grad_W2, grad_W20, grad_W3, grad_W30.T]
    return grads

# Weight updates using stochastic gradient descent
def update(theta, gradients):
    W1 = theta[0] - learn_rate*gradients[0]
    W10 = theta[1] - learn_rate*gradients[1]
    W2 = theta[2] - learn_rate*gradients[2]
    W20 = theta[3] - learn_rate*gradients[3]
    W3 = theta[4] - learn_rate*gradients[4]
    W30 = theta[5] - learn_rate*gradients[5]
    theta_new = [W1, W10, W2, W20, W3, W30]
    return theta_new
    
    
# load training & testing data
train_data_path = 'train_data' # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
test_data_path = 'test_data' # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
train_data_root = pathlib.Path(train_data_path)
test_data_root = pathlib.Path(test_data_path)

# list all training images paths，sort them to make the data and the label aligned
all_training_image_paths = list(train_data_root.glob('*'))
all_training_image_paths = sorted([str(path) for path in all_training_image_paths])

# list all testing images paths，sort them to make the data and the label aligned
all_testing_image_paths = list(test_data_root.glob('*'))
all_testing_image_paths = sorted([str(path) for path in all_testing_image_paths])

# load labels
training_labels = np.loadtxt('labels/train_label.txt', dtype = int) # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
# convert 1-10 to 0-9 and build one hot vectors
training_labels = tf.reshape(tf.one_hot(training_labels - 1 , 10, dtype=tf.float32), (-1, 10))
testing_labels = np.loadtxt('labels/test_label.txt', dtype = int) # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
testing_labels = tf.reshape(tf.one_hot(testing_labels - 1 , 10, dtype=tf.float32), (-1, 10))

# load images
training_set = load_images(all_training_image_paths)
testing_set = load_images(all_testing_image_paths)

# Train model
iters = 1
train_loss = list()        # create list to store training losses
test_loss = list()         # create list to store testing losses
train_accuracy = list()    # create list to store training accuracy
test_accuracy = list()     # create list to store training accuracy
train_error_per_digit = list()   # create list to store classification error per digit
test_error_per_digit = list()
while (iters<epochs+1):
    # create batches for input
    batch = list(range(training_set.shape[0]))
    batch_n = r.sample(batch, batch_size)
    X_batch = tf.gather(training_set, batch_n, axis=0)
    Y_batch = tf.gather(training_labels, batch_n, axis=0)
    # Perform forward propagation
    Y_hat, H_1, H_2 = forward_prop(X_batch, theta)
    # Compute training loss and accuracy
    train_l= loss(Y_hat, Y_batch)
    train_loss.append(train_l.numpy())
    Y_hat = tf.one_hot(tf.math.argmax(Y_hat, 1), 10, dtype=tf.float32)
    train_acc = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(Y_batch, Y_hat), axis=1))
    train_accuracy.append(train_acc.numpy())
    # Compute training classification error for each digit
    for i in range(10):
        y_hat = tf.reduce_sum(tf.math.multiply(Y_hat[:, i], Y_batch[:, i]))
        label = tf.reduce_sum(Y_batch[:, i])
        error = 1- y_hat/label
        train_error_per_digit.append(error)
    # Calculate gradients
    Gradient = gradients(X_batch, Y_hat, Y_batch, theta, H_2, H_1)
    # Update weights using gradients
    theta = update(theta, Gradient)
    # Compute testing accuracy and loss
    Y_hat_test, H1, H2 = forward_prop(testing_set, theta)
    test_l = loss(Y_hat_test, testing_labels)
    test_loss.append(test_l.numpy())
    Y_hat_test = tf.one_hot(tf.math.argmax(Y_hat_test, 1), 10, dtype=tf.float32)
    test_acc = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(testing_labels, Y_hat_test), axis=1))
    test_accuracy.append(test_acc.numpy())
    # Compute testing classification error for each digit
    for i in range(10):
        y_hat_test = tf.reduce_sum(tf.math.multiply(Y_hat_test[:, i], testing_labels[:, i]))
        labels = tf.reduce_sum(testing_labels[:, i])
        error = 1- y_hat_test/labels
        test_error_per_digit.append(error)
    iters += 1
    
# Plot testing and training losses and accuracy
x= list(range(epochs))
plt.plot(x,train_loss)
plt.ylabel('Loss')
plt.title('Training loss')
plt.show()

plt.plot(x,train_accuracy)
plt.ylabel('Accuracy')
plt.title('Training accuracy')
plt.show()

plt.plot(x,test_accuracy)
plt.ylabel('Accuracy')
plt.title('Testing accuracy')
plt.show()

plt.plot(x,test_loss)
plt.ylabel('Loss')
plt.title('Testing loss')
plt.show()

# Plot classification error per digit
plt.style.use('seaborn')
index = [i for i in range(epochs)]
train_error_per_digit = np.reshape(train_error_per_digit, [epochs, -1, 10])
test_error_per_digit = np.reshape(test_error_per_digit, [epochs, -1, 10])
plt.xlabel('Epochs')
plt.ylabel('Classification Error')
for digit in range(10):
    plt.plot(index, train_error_per_digit[:, 0, digit])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize='medium')
plt.show()

plt.xlabel('Epoch')
plt.ylabel('Classification Errors')
for digit_index in range(10):
    plt.plot(index, test_error_per_digit[:, digit_index])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize='medium')
ax = plt.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
plt.show()

# Save parameters
Theta = list()
for param in theta:
    a = tf.convert_to_tensor(param)
    Theta.append(a)
filehandler = open("nn_parameters.txt", "wb")
pickle.dump(Theta, filehandler, protocol=2)
filehandler.close()
    
    
