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
from keras.preprocessing import image
from pylab import imread,subplot,imshow,show
import Util

model = load_model("CNN2.h5") 

image = cv2.imread("testsvm.jpg")
im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
_,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]
cont_sort_area=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)[2:10]
new_conts=Util.extract_cont_row(cont_sort_area)
Util.print_info(new_conts)
dr = cv2.drawContours(image,new_conts,-1,(000,000,255),2)
images=[]
pred=[]
for i in new_conts:
    (x,y,w,h) = cv2.boundingRect(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
    roi = thre[y:y+h,x:x+w]
    plt.imshow(roi)
    plt.show()
    roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
    plt.imshow(roi)
    plt.show()
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    plt.imshow(roi)
    plt.show()
    roi = cv2.dilate(roi, (3, 3)) 
    plt.imshow(roi)
    plt.show()
    images.append(roi)
    # Calculate the HOG features
    # Util.print_info(roi_hog_fd,'true')
    # plt.plot(roi_hog_fd)
    # plt.show()
    roi =np.expand_dims(roi,axis=3)
    roi =np.expand_dims(roi,axis=0)
    nbr = model.predict(roi)
    # cv2.putText(image, str(int(nbr[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    # cv2.namedWindow('image1111',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image1111', 700,700)
    # cv2.imshow("image1111",image)
    pred.append(nbr[0])

print(pred)
print(np.argmax(pred,axis=1))
