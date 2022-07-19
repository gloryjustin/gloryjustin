# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:18:00 2022
Programming assignement 1 for Deep learning class
This is a code to perform multi class classification on a set of handwritten images
of 5 numbers. This is implemented using multi-class logistic regression. The weights 
were initialied and their parameters optimized using Stochastic Gradient Descent

@author: glory justin
"""
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random as r
import tensorflow as tf

# read images
def load_images(path_list):
    number_samples = len(path_list)
    Images = []
    for each_path in path_list:
        img = plt.imread(each_path)
        # divided by 255.0
        img = img.reshape(784, 1) / 255.0
        '''
        In some cases data need to be preprocessed by subtracting the mean value of the data and divided by the 
        standard deviation to make the data follow the normal distribution.
        In this assignment, there will be no penalty if you don't do the process above.
        '''
        # add bias
        img = np.vstack((img, [1]))
        Images.append(img)
    data = tf.convert_to_tensor(np.array(Images).reshape(number_samples, 785), dtype=tf.float32)
    return data

# loss function
def loss(X,Y,theta):
    lambd=2e-2
    # multiply data X with weights theta
    X = X.numpy()
    theta= theta.numpy()
    Y_t = tf.transpose(Y)
    Y= Y.numpy()
    N= X.shape[0]
    Z = X @ theta
    # loss function based on cross entropy plus L2 regularization
    L_NLL = 1/N *(np.trace(X @ theta @ Y_t.numpy())+np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    L2 = lambd*np.linalg.norm(theta)
    L_NLL +=L2
    return L_NLL

# Stochastic Gradient Descent
def SGD(X,Y,theta):
    lambd=1e-2    # L2 regularization constant
    neta = 1e-2   # learn rate
    batch = 100    # batch size
    # Generate random batches
    b= X.numpy().shape[0]
    a = list(range(b))
    batch_X = tf.gather(X, (r.sample(a, batch)), axis=0)
    batch_Y = tf.gather(Y, (r.sample(a, batch)), axis=0)
    batch_X_t = tf.transpose(batch_X)
    batch_Z = batch_X.numpy() @ theta.numpy()
    # Softmax function
    Z_soft = np.exp(batch_Z)/np.sum(np.exp(batch_Z), axis=0)
    # gradient with L2 regularization
    grad = (1/batch)*(batch_X_t.numpy() @ (batch_Y.numpy() - Z_soft)) + 2*lambd*theta.numpy()
    # parameter update
    theta = theta.numpy()
    theta += - neta*grad
    return tf.Variable(theta), tf.Variable(batch_X), tf.Variable(batch_Y), tf.Variable(batch_Z)
    
    
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
# convert 1-5 to 0-4 and build one hot vectors
training_labels = tf.reshape(tf.one_hot(training_labels - 1 , 5, dtype=tf.float32), (-1, 5))
testing_labels = np.loadtxt('labels/test_label.txt', dtype = int) # Make sure folders and your python script are in the same directory. Otherwise, specify the full path name for each folder.
testing_labels = tf.reshape(tf.one_hot(testing_labels - 1 , 5, dtype=tf.float32), (-1, 5))

# load images
training_set = load_images(all_training_image_paths)
testing_set = load_images(all_testing_image_paths)


# Train model
theta = tf.Variable(0.1*np.random.rand(785,5)) # Initialize weights using random variables
max_iter = 2
iter = 0
training_loss = list()
training_accuracy=list()
test_accuracy= list()
test_loss = list()
while (iter<max_iter):
    X = load_images(all_training_image_paths)  # load data
    Y = training_labels                        # get labels
    # Update parameters and get inputs for loss function
    theta, X, Y, Z = SGD(X,Y,theta)
    # get loss  
    train_loss= loss(X,Y,theta)
    # Store training loss for plotting
    training_loss.append(train_loss)
    # Get predicted labels
    y_hat = tf.one_hot(tf.math.argmax(Z, 1), 5, dtype=tf.float32)
    # Compare predicted values with actual values to obtain accuracy
    train_accuracy= tf.reduce_mean(tf.reduce_sum(tf.math.multiply(Y, y_hat), axis=1))
    train_accuracy = train_accuracy.numpy()
    # Store accuracy for plotting
    training_accuracy.append(train_accuracy)
    # Get testing loss and accuracy
    Z_test = testing_set.numpy() @ theta.numpy()
    Z_test_soft = np.exp(Z_test)/np.sum(np.exp(Z_test), axis=0) #softmax
    Z_test_soft = tf.Variable(Z_test_soft)
    y_hat_test= tf.one_hot(tf.math.argmax(Z_test_soft, 1), 5, dtype=tf.float32)
    test_los = loss(testing_set, testing_labels, theta)
    test_loss.append(test_los)
    test_acc= tf.reduce_mean(tf.reduce_sum(tf.math.multiply(testing_labels, y_hat_test), axis=1))
    test_accuracy.append(test_acc.numpy())
    iter +=1

# Classification error
for i in range(5):
    predict = tf.reduce_sum(tf.math.multiply(y_hat_test[:, i], testing_labels[:, i]))
    labels = tf.reduce_sum(testing_labels[:, i])
    error = 1- predict/labels
    print(error)
    
# Plot testing and training accuracy and loss
x= list(range(max_iter))
plt.plot(x,training_loss)
plt.ylabel('Loss')
plt.title('Training loss')
plt.show()

plt.plot(x,training_accuracy)
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


# Save learned parameters
filehandler = open("multiclass_parameters.txt","wb")
pickle.dump(theta, filehandler)
filehandler.close()

# Visualize weights
theta = theta.numpy()
Img1 = theta[1:,0].reshape(28,28)
plt.imshow(Img1)
plt.colorbar()
plt.show()
Img1 = theta[1:,1].reshape(28,28)
plt.imshow(Img1)
plt.colorbar()
plt.show()
Img1 = theta[1:,2].reshape(28,28)
plt.imshow(Img1)
plt.colorbar()
plt.show()
Img1 = theta[1:,3].reshape(28,28)
plt.imshow(Img1)
plt.colorbar()
plt.show()
Img1 = theta[1:,4].reshape(28,28)
plt.imshow(Img1)
plt.colorbar()
plt.show()



    













