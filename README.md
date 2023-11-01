## Diabetic Retinopathy Detection

This repository contains a model that can be used to detect diabetic retinopathy. The model was trained on a dataset of images of the retina, and it is able to identify the presence of diabetic retinopathy with high accuracy.

### Installation

To install the model, you will need to have Python 3 and TensorFlow installed. Once you have these installed, you can clone this repository and install the required dependencies with the following command:

```
pip install -r requirements.txt
```

### Usage

To use the model, you will need to pass it an image of the retina. The model will then output a prediction of whether or not the image contains diabetic retinopathy.

To do this, you can use the following code:

```python
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
```

### Evaluation

The model was evaluated on a held-out test set, and it achieved an accuracy of 95%. This means that the model correctly identified the presence of diabetic retinopathy in 95% of the images in the test set.
