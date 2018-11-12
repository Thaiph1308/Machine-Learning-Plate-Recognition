import cv2
import numpy as np
from keras.models import load_model
import Util
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

img_width = 28
img_height = 28

model=load_model("CNN.h5")
model.summary()
for layer in model.layers:
    print(layer.get_input_at(0).get_shape().as_list())
im=cv2.imread("digit.jpg")
im=cv2.resize(im,(img_height,img_width))
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(np.shape(im))
arr = np.reshape(im,(img_height,img_width))
print(np.shape(arr))
arr = np.expand_dims(arr,axis=0)
arr = np.expand_dims(arr,axis=0)
print(np.shape(arr))
print(model.predict(arr))
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(np.shape(X_train))
# print(model.predict(X_test[0]))