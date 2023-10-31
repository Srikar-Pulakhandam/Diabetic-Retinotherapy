import numpy as np
import tensorflow as tf

from model import DiabeticRetinopathyDetector

# Load the model
model = DiabeticRetinopathyDetector()

# Load the image
image = tf.io.read_file('image.jpg')
image = tf.image.decode_jpeg(image)

# Normalize the image
image = tf.image.convert_image_dtype(image, tf.float32)
image = tf.image.resize(image, (224, 224))

# Make a prediction
prediction = model.predict(image)

# Print the prediction
print(prediction)
