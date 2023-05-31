import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten


#get data
train_path = 'resources/trainMain'
img_size = 224
epochs = 12
batch_size = 16

#dataset extension
train_datagen = ImageDataGenerator(width_shift_range = 0.001,
                                    rotation_range=10,
                                    height_shift_range = 0.001,
                                    horizontal_flip = True,
                                    rescale = 1./255,
                                    validation_split = 0.2)

validation_datagen = ImageDataGenerator(rescale = 1./255,
                                        validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(directory = train_path,
                                                              target_size=(img_size, img_size),
                                                              batch_size=batch_size,
                                                              color_mode="grayscale",
                                                              class_mode="categorical",
                                                              subset="training")

validation_generator = validation_datagen.flow_from_directory(directory = train_path,
                                                                    target_size=(img_size, img_size),
                                                                    batch_size=batch_size,
                                                                    color_mode="grayscale",
                                                                    class_mode="categorical",
                                                                    subset="validation")

#model architecture
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.15),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.1),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.1),

    Dense(256, activation='relu'),
    Dropout(0.15),

    Dense(4, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

#start of training
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=validation_generator)

score = model.evaluate(validation_generator)
score = 100 * round(score[1], 3)
print(f"The accuracy of this model is {score}%")

#visualisation of results
plt.plot(history.history['accuracy'],
         label='Точность на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Точность на проверочном наборе')
plt.plot(history.history['loss'],
         label='Доля потерь на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Доля потерь на проверь наборе')

plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

model.save("Model_2.2.h5")