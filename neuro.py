# -*- coding: utf-8 -*-
"""Neuro.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16oKDAUPgkI6Hl_qlFNrDc2ZIBj3ys4Lp
"""

from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.utils.image_dataset import image_dataset_from_directory
from keras import layers

img_side = 48
batch_size = 19
epochs = 15
train_dir = '/content/train'
test_dir = '/content/test'

#Dataset identification
train_dataset = tf.keras.utils.image_dataset_from_directory(
                                          train_dir,
                                          seed = 123,
                                          image_size = (img_side, img_side),
                                          batch_size = batch_size,
                                          color_mode = "grayscale")

test_dataset = tf.keras.utils.image_dataset_from_directory(   
                                          test_dir,
                                          seed = 123,
                                          image_size = (img_side, img_side),
                                          batch_size = batch_size,
                                          color_mode = "grayscale")

#Architecture of model

model = Sequential([
    layers.Conv2D(19, (3,3), padding='same', activation='relu', input_shape=(48,48,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(38, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(38, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(76,activation='relu'),
    layers.Dense(7, activation='softmax')
])

#Model compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

training = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

plt.plot(training.training['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(training.training['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.plot(training.training['loss'],
         label='Доля loss')
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)

plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

from google.colab import files
uploaded = files.upload()