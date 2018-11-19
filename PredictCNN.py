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
import Bsx
import os
img_width = 28
img_height = 28

#model = load_model("E:\\IoTProject\\Machine-Learning-Plate-Recognition\\CNN2.h5") 
model = load_model("CNN2.h5")
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
        Util.serch_number_bouding_rect(contours)
        #sorteddata = sorted(zip(rects_area, contours), key=lambda x: x[0], reverse=True)
        cont_sort_area=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)[1:10]
        (new_conts,BoudingBoxes)=Util.extract_cont_row(cont_sort_area,True)
        #print(new_conts)
        return (image,new_conts,BoudingBoxes)

def full_predict(imagepath):
        (image,conts,BoudingBoxes) = extract_contour(imagepath)
        cv2.drawContours(image, conts, -1, (0, 255, 0), 3)
        plt.imshow(image)
        plt.show()
        Character=[]
        Areas = []
        for i,box in enumerate(BoudingBoxes):
                (x,y,w,h) = box
                #Util.print_info(image)
                Areas.append(w*h)
                img = image[y:y+h,x:x+w]
                im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
                (thresh, im_bw) = cv2.threshold(im_blur, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                im_bw=np.pad(im_bw,(20,20),'constant',constant_values=(0,0)) 
                img_blur_resize=cv2.resize(im_bw,(28,28),interpolation=cv2.INTER_AREA)        
                # plt.imshow(img_blur_resize)
                # plt.show()      
                cv2.imwrite("c.jpg",img_blur_resize)
                # # plt.imshow(img)
                # # plt.show()
                Character.append(img_blur_resize)
                #print(Character)
        print(Areas)
        return Character
Character_images = full_predict(Bsx.Extract_Plate("bxx3.jpg","test.jpg"))    
z=np.expand_dims(Character_images,axis=3)
y=model.predict(z)
y_true = np.argmax(y,axis=1)
y_str= "".join(str(x) for x in y_true)        
print(y_str)
# def services():      
#         path= "E:\\testIOT\\Server\\StoreImages"
#         for i,filename in enumerate(os.listdir(path)):
#                 print(filename)
#                 Character_images = full_predict(Bsx.Extract_Plate("E:\\testIOT\\Server\\StoreImages\\"+ filename,"test.jpg"))    
#                 z=np.expand_dims(Character_images,axis=3)
#                 y=model.predict(z)
#                 y_true = np.argmax(y,axis=1)
#                 y_str= "".join(str(x) for x in y_true)
#                 print(y_str)
#                 with open("E:\\testIOT\\Server\\StoreTxt\\%d.txt" %(i+1),"w") as text_file:
#                         text_file.write(y_str)
#                 os.remove("E:\\testIOT\\Server\\StoreImages\\"+ filename)


#Character_images = full_predict(Bsx.Extract_Plate("E:\\testIOT\\Server\\StoreImages\\bxx.jpg","test.jpg"))
#Util.sub_plot(Character_images,5,2)
# image: list of character
#Util.print_info(images[0])
#Util.print_info(np.asarray(images))
#Util.print_info(Character_images)
#Util.sub_plot(images,10,1)
# img = image.load_img(path="test.jpg",grayscale=True,target_size=(28,28,1))
# img = image.img_to_array(img)
# List_of_images = []
# for image in Character_images:
#     x=Util.image_reshape_2(image)
#     List_of_images.append(x)
# Util.print_info(List_of_images[0])
#get_model_info(model)
# plt.imshow(List_of_images[0].squeeze(),aspect="auto")
# plt.show()
# z=np.expand_dims(Character_images[0],axis=3)
# y=model.predict(np.expand_dims(z,axis=0))
######
# z=np.expand_dims(Character_images,axis=3)
# y=model.predict(z)
# y_true = np.argmax(y,axis=1)
# y_str= "".join(str(x) for x in y_true)
# # print(y)
# # print(y_true)
# print(y_str)
# with open("E:\\testIOT\\Server\\StoreTxt\\output.txt","w") as text_file:
#         text_file.write(y_str)
########
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