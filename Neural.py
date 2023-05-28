import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten


#get data
train_path = 'resources/trainMain'
img_size = 224
epochs = 8
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
                                                              subset="training"
                                                              )

validation_generator = validation_datagen.flow_from_directory(directory = train_path,
                                                                    target_size=(img_size, img_size),
                                                                    batch_size=batch_size,
                                                                    color_mode="grayscale",
                                                                    class_mode="categorical",
                                                                    subset="validation"
                                                                    )

#model architecture
model= Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=['accuracy']
)


#start of training
training = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

#visualisation results
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

model.save("Model_7.3")