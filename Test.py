import keras.models
from keras.preprocessing.image import image_utils
import numpy as np
import matplotlib.pyplot as plt
import os
classes = os.listdir('resources/trainMain')

IMAGE_SIZE = 224
BATCH_SIZE = 8

model = keras.models.load_model("Model_2.4", compile=False)
x = 0
path = r"resources/testNeural"

for i in os.listdir(path):
    img = image_utils.load_img(f"resources/testNeural/{i}", color_mode="grayscale", target_size=(IMAGE_SIZE, IMAGE_SIZE))

    image_tensor = image_utils.img_to_array(img)
    image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor /= 255

    prediction = model.predict(image_tensor)

    plt.subplots(1, 1)
    plt.imshow(img)
    plt.xlabel(classes[np.argmax(prediction)])
    plt.grid(False)
    x += 1

plt.show()