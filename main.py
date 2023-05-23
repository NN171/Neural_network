import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

img_side = 128
batch_size = 43
epochs = 22
train_dir = 'resources/trainMain'
test_dir = 'resources/test'

#Dataset identification
train_dataset = tf.keras.utils.image_dataset_from_directory(
                                          train_dir,
                                          image_size=(img_side, img_side),
                                          batch_size=batch_size,
                                          color_mode="grayscale")

test_dataset = tf.keras.utils.image_dataset_from_directory(
                                          test_dir,
                                          image_size=(img_side, img_side),
                                          batch_size=batch_size,
                                          color_mode="grayscale")

#Architecture of model

model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

#Model compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

training = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

plt.plot(training.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(training.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.plot(training.history['loss'],
         label='Доля loss')
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)

plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()