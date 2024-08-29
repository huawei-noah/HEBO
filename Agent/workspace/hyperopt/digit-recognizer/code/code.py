#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install keras-tuner')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# from kerastuner import RandomSearch

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from keras.callbacks import ReduceLROnPlateau
# from keras.optimizers import RMSprop
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.datasets import mnist
FILE_PATH = "./workspace/hyperopt/digit-recognizer/data/"

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Load the data
train = pd.read_csv(FILE_PATH+'train.csv')
labels = train.iloc[:,0].values.astype('int32')

X_train = (train.iloc[:,1:].values).astype('float32')
X_test = (pd.read_csv(FILE_PATH+'test.csv').values).astype('float32')

#reshape into images
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# one hot encoding
y_train = tf.keras.utils.to_categorical(labels) 

# print("Check data")
# print(labels)
# print(X_train[0].shape)
# print(y_train)


# Load Data from Keras MNIST
(train_imagesRaw, train_labelsRaw), (test_imagesRaw, test_labelsRaw) = mnist.load_data()


#reshape into images
X_train_keras = train_imagesRaw.reshape(-1,28,28,1)
X_test_keras = test_imagesRaw.reshape(-1,28,28,1)

# print("X_train_keras",X_train_keras.shape)
# print("X_test_keras",X_test_keras.shape)

train_labels_keras = tf.keras.utils.to_categorical(train_labelsRaw)
test_labels_keras = tf.keras.utils.to_categorical(test_labelsRaw)
# print("train_labels_keras ",train_labels_keras.shape)
# print("test_labels_keras ", test_labels_keras.shape)


# merge datasets

train_images = np.concatenate((X_train_keras,X_train,X_test_keras), axis=0)
# print("new Concatenated train_images ", train_images.shape)
# print("_"*50)

train_labels = np.concatenate((train_labels_keras,y_train,test_labels_keras), axis=0)
# print("new Concatenated train_labels ", train_labels.shape)


#visualize an image

# fig = plt.figure()
# plt.imshow(X_train[6][:,:,0], cmap='gray', interpolation='none')
# plt.xticks([])
# plt.yticks([])


scale = np.max(train_images)
train_images /= scale
X_test /= scale

#visualize scales

# print("Max: {}".format(scale))


# X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.10)


# # Here we define the input and output layer sizes
input_size = X_train.shape
n_logits = y_train.shape[1]

# print("Input: {}".format(input_size))
# print("Output: {}".format(n_logits))

num_layers = 8 #hp.Int('num_layers', min_value=2, max_value=16, step=2)

lr = 1e-4 #hp.Choice('learning_rate', [1e-3, 5e-4])
filters = 128 #hp.Int('filters_' + idx, 32, 256, step=32, default=64)
pool_type = 'max' #hp.Choice('pool_' + idx, values=['max', 'avg'])

inputs = layers.Input(shape=(28, 28, 1))
x = inputs
for idx in range(num_layers):
    idx = str(idx)
    x = layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                        activation='relu')(x)

    # add a pooling layers if needed
    if x.shape[1] >= 8:
        if pool_type == 'max':
            x = layers.MaxPooling2D(2)(x)
        elif pool_type == 'avg':
            x = layers.AveragePooling2D(2)(x)

# My dense layer

x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(n_logits, activation='softmax')(x)
            
# Build model
model = keras.Model(inputs, outputs)
model.compile(optimizer=Adam(lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])




# def build_model(hp):
#     """Function that build a TF model based on hyperparameters values.
#     Args:
#         hp (HyperParameter): hyperparameters values
#     Returns:
#         Model: Compiled model
#     """
#     num_layers = hp.Int('num_layers', min_value=2, max_value=16, step=2)
    
#     lr = hp.Choice('learning_rate', [1e-3, 5e-4])

#     inputs = layers.Input(shape=(28, 28, 1))
#     x = inputs

#     for idx in range(num_layers):
#         idx = str(idx)

#         filters = hp.Int('filters_' + idx, 32, 256, step=32, default=64)
#         x = layers.Conv2D(filters=filters, kernel_size=3, padding='same',
#                           activation='relu')(x)

#         # add a pooling layers if needed
#         if x.shape[1] >= 8:
#             pool_type = hp.Choice('pool_' + idx, values=['max', 'avg'])
#             if pool_type == 'max':
#                 x = layers.MaxPooling2D(2)(x)
#             elif pool_type == 'avg':
#                 x = layers.AveragePooling2D(2)(x)

#     # My dense layer
    
#     x = layers.Flatten()(x)
#     x = layers.Dense(256, activation='relu')(x)
#     x = layers.Dense(256, activation='relu')(x)
#     x = layers.Dense(256, activation='relu')(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(n_logits, activation='softmax')(x)
              
#     # Build model
#     model = keras.Model(inputs, outputs)
#     model.compile(optimizer=Adam(lr),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=8,
#     executions_per_trial=3,
#     directory='my_dir',
#     project_name='mnist')

# tuner.search_space_summary()


# tuner.search(X_train, y_train,
#              epochs=30,
#              validation_data=(X_val, y_val))


# model = tuner.get_best_models(num_models=1)[0]
# model.summary()


# # generate predictions
# predictions_vector = model.predict(X_test, verbose=0)
# predictions = np.argmax(predictions_vector,axis=1)

# pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions}).to_csv("preds.csv", index=False, header=True)

