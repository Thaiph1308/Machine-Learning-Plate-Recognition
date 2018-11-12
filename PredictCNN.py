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

model=load_model("CNN.h5")
model.summary()
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(np.shape(X_train))
# print(model.predict(X_test[0]))