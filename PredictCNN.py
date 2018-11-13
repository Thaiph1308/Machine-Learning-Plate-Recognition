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
from keras.preprocessing.image import img_to_array, load_img
import Util

img_width = 28
img_height = 28
model = load_model("CNN.h5") 

# def init_cnn_with_weight(weightpath):
#     model=load_model("CNN.h5")
#     return model

def get_model_info(model):
    model.summary()
    for layer in model.layers:
        print(layer.get_input_at(0).get_shape().as_list())

def predict(image):
    return np.argmax(model.predict(image),axis=1)

def extract_contour(image):
    image = cv2.imread(image)
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
    im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
    _,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    cont_sort_area=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)[2:10]
    (new_conts,BoudingBoxes)=Util.extract_cont_row(cont_sort_area,True)
    #print(new_conts)
    return (image,new_conts,BoudingBoxes)

def full_predict(imagepath):
    (image,conts,BoudingBoxes) = extract_contour(imagepath)
    Character=[]
    for box in BoudingBoxes:
        (x,y,w,h) = box
        Util.print_info(image)
        img = image[y:y+h,x:x+w]
        Character.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    print(Character)
    return Character

images = full_predict("testsvm.jpg")
# image: list of character
Util.print_info(images[0])
Util.print_info(np.asarray(images))
Util.print_info(images)
#Util.sub_plot(images,10,1)
List_of_images = []
for image in images:
    x=Util.image_reshape_2(image)
    List_of_images.append(x)
Util.print_info(List_of_images[0])
#get_model_info(model)
y=model.predict(np.asarray(List_of_images))
y_true = np.argmax(y,axis=1)
print(y_true)
#Util.sub_plot(images,10,1)
# print(np.shape(Util.image_reshape("digit.jpg")))
# print(model.predict)
# print(predict(np.expand_dims(Util.image_reshape("digit.jpg"),axis=0)))
# im=cv2.imread("digit.jpg")
# im=cv2.resize(im,(img_height,img_width))
# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# print(np.shape(im))
# arr = np.reshape(im,(img_height,img_width))
# print(np.shape(arr))
# arr = np.expand_dims(arr,axis=0)
# arr = np.expand_dims(arr,axis=0)
# print(np.shape(arr))
# y = model.predict(arr)
# Y_true = np.argmax(y, axis = 1) 
# print(Y_true)
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(np.shape(X_train))
# print(model.predict(X_test[0]))