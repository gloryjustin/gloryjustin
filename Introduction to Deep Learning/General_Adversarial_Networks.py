# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:27:33 2022
Deep learning class final project. 

This project involves generating images of certain celebrities' faces uses a 
conditional General Adversarial Network (cGAN). These faces are required to 
have certain attributes among 5.  

This network comprises of two components. There is a generator and a discriminator. 
The generator takes in noise and desired attributes and outputs a synthetic image. 
While the discriminator takes in real images and fake images as well as their 
attributes and classifies them as real or fake.
s
The images generated are evaluated using the Inception score and FID.s

@author: Glory
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt

# Define hyper parameters
batch = 50  # batch size
lr = 1e-3   # learn rate
epochs = 200  # number of epochs
num_classes = 5
x_shape=(64,64,3)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

# First load dataset
training_data = np.load('images.npy')   # 202,599 x 64 x 64 x 3
training_data = training_data[0:100000, :, :, :]
training_label = np.load('attributes5.npy')
training_label = training_label[0:100000, :]

# Create training bathches
train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_label))
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(50)

# Define Generator network using the functional API in Keras
def Generator():
    y_in = layers.Input((5,))
    y = layers.Embedding(2,50,input_length=5)(y_in)
    y = layers.Flatten()(y)
    y = layers.Dense(64)(y)
    y = layers.Reshape((8,8,1))(y)
    z_in = layers.Input((100,))
    z = layers.Dense(128*8*8)(z_in)
    z= layers.LeakyReLU(alpha=0.2)(z)
    z = layers.Reshape((8,8,128))(z)
    yz = layers.Concatenate()([y,z])
    x_hat = layers.Conv2DTranspose(128, kernel_size=4, strides=2)(yz)
    x_hat = layers.LeakyReLU(alpha=0.2)(x_hat)
    x_hat = layers.Conv2DTranspose(128, kernel_size=4, strides=2)(x_hat)
    x_hat = layers.LeakyReLU(alpha=0.2)(x_hat)
    x_hat = layers.Conv2DTranspose(128, kernel_size=2, strides=2)(x_hat)
    x_hat = layers.LeakyReLU(alpha=0.2)(x_hat)
    # x_hat = layers.Conv2DTranspose(128, kernel_size=2, strides=2)(x_hat)
    # x_hat = layers.LeakyReLU(alpha=0.2)(x_hat)
    output = layers.Conv2D(3, kernel_size=8, activation='tanh')(x_hat)
    Gen = models.Model([z_in, y_in], output)
    return Gen

# Define Discriminator
def Discriminator():
    y_in = layers.Input((5,))
    y = layers.Embedding(2, 50, input_length=5)(y_in)
    y = layers.Flatten()(y)
    y = layers.Dense(64*64)(y)
    y = layers.Reshape((64,64,1))(y)
    x_in = layers.Input((x_shape))
    xy = layers.Concatenate()([x_in,y])
    y_hat = layers.Conv2DTranspose(128, kernel_size=3, strides=2)(xy)
    y_hat = layers.LeakyReLU(alpha=0.2)(y_hat)
    y_hat = layers.Conv2DTranspose(256, kernel_size=3, strides=2)(y_hat)
    y_hat = layers.LeakyReLU(alpha=0.2)(y_hat)
    y_hat = layers.Conv2DTranspose(512, kernel_size=3, strides=2)(y_hat)
    y_hat = layers.LeakyReLU(alpha=0.2)(y_hat)
    y_hat = tf.keras.layers.Flatten()(y_hat)
    output = layers.Dense(1, activation='sigmoid')(y_hat)
    Dis = models.Model([x_in, y_in], output)
    Dis.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    return Dis

# Combine Generator and Discriminator to form GAN
def GAN(Gen, Dis):
    Dis.trainable = False
    z_in, y_in = Gen.input
    x_hat = Gen.output
    pred = Dis([x_hat, y_in])
    GAN = models.Model([z_in, y_in], pred)
    GAN.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy())
    return GAN

# Define metrics, Inception score and FID
def metrics(x, x_hat):
    # Resize images for input into inception model
    x = tf.image.resize(x, [299, 299])
    x_hat = tf.image.resize(x_hat, [299, 299])
    x = preprocess_input(x)
    x_hat = preprocess_input(x_hat)
    # Create inception model
    inception_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape=(299,299,3), classes=5)
    # Get outputs from inception model
    y = inception_model(x)
    y_hat = inception_model(x_hat)
    # Get mean and covariance for FID
    mean_1 = tf.reduce_mean(y, axis = 0)
    mean_2 = tf.reduce_mean(y_hat, axis = 0)
    cov_1 = np.cov(y)
    cov_2 = np.cov(y_hat)
    # Compute FID
    FID = tf.reduce_sum((mean_1-mean_2)**2) + np.trace(cov_1+cov_2 - 2*np.real(np.sqrt(cov_1@cov_2)))
    # Compute Inception Score
    Score = []
    # frist get class probabilities
    for i in range(5):
        i_1, i_2 = i*int(batch/5), i*int(batch/5)+int(batch/5)
        pyx = y_hat[i_1:i_2]
        py = np.expand_dims(pyx.mean(axis=0), 0)
        kl = pyx * (tf.math.log(pyx + 1e-16) - tf.math.log(py + 1e-16))
        kl_mean = tf.reduce_mean(kl, axis=1)
        Score.append(kl_mean.numpy())
    IS = tf.reduce_mean(tf.math.exp(Score))  # then get the average of the exponent of the KL divergence
    return FID, IS

# Train GAN model
Gen = Generator()
Gen.summary()
Dis = Discriminator()
Dis.summary()
Gan = GAN(Gen, Dis)
Gan.summary()
epoch = 0
# Create lists to store training metrics
train_IS = list()
train_FID = list()
Gan_loss = list()
Dis_loss = list()
while epoch < epochs:
    epoch += 1
    print('Start training: Epoch', epoch)
    temp_gan = list()
    temp_dis = list()
    for x, y in train_dataset:
        # Generate random gaussian noise
        z = tf.random.normal(shape=(batch, 100))
        x = tf.divide(tf.cast(x, tf.float32), 255.0)
        x_hat = Gen.predict([z,y])
        pred = Dis.predict([x_hat, y])
        label_1 = tf.zeros([batch,1])  # labels for fake images
        label_2 = tf.ones([batch,1])   # labels for real images
        loss_dis, acc = Dis.train_on_batch([x, y], label_2)
        loss_dis2, acc = Dis.train_on_batch([x_hat, y], label_1)
        loss_gan = GAN.train_on_batch([z, y], pred)
        temp_dis.append((loss_dis.numpy()+loss_dis2.numpy())/2)
        temp_gan.append(loss_gan.numpy())
    Gan_loss.append(sum(temp_gan)/len(temp_gan))
    Dis_loss.append(sum(temp_dis)/len(temp_dis))
    # Get metrics
    FID, IS = metrics(x, x_hat)
    # Append metrics to lists for storage
    train_IS.append(IS)
    train_FID.append(FID)
    print('Epoch:', epoch, 'GAN loss:', Gan_loss[-1], 'Discriminator loss', Dis_loss[-1],
          'Inception score', train_IS[-1], 'FID', train_FID[-1])
    
plt.imshow(x_hat[0,:,:,:])