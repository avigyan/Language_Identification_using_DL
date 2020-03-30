# USAGE
# python test_network.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse

from scipy.misc import imread

# load the image
image = imread('kan.png', 'L')
#image = imread('tam.png', 'L')
#image = imread('tel.png', 'L')

orig = image.copy()
# Image shape should be (cols, rows, channels)
if len(orig.shape) == 2:
	orig = np.expand_dims(orig, -1)

assert len(orig.shape) == 3

orig=np.divide(orig, 255.0)  # Normalize images to 0-1.0
orig = np.expand_dims(orig, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('weights.21.model')

# classify the input image
predictions = model.predict(orig)[0]

print("*****************")
label=np.argmax(predictions)

if label == 0:
	print("Language detected: Kannada")
elif label == 1:
	print("Language detected: Tamil")
elif label == 2:
	print("Language detected: Telugu")

