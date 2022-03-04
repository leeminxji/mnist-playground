import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("./model.h5")

# image ==> np.array
size = 28, 28

img = Image.open("./img/6.png").resize(size, Image.ANTIALIAS).convert('L') # NOTE: example

np_img = np.array(img)
np_img = np_img/255
np_img = np.reshape(np_img, (1, 28, 28, 1))

# prediction
predictions = model.predict(np_img)

# result
print(np.argmax(predictions[0]))
