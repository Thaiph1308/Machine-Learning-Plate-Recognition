import cv2
import pickle
import numpy as np
from skimage.feature import hog
from keras.models import load_model
import Util
import matplotlib.pyplot as plt

with open('SVM90.pkl', 'rb') as fid:
    model = pickle.load(fid)
image = cv2.imread("testsvm.jpg")
im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
_,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]

cont_sort_area=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)[2:11]
# (sort_cont,boundingBoxes) = Util.sort_contours(s)
# z= [cv2.boundingRect(c) for c in contours]
# x= [cv2.boundingRect(c) for c in s]
# z1= [cv2.contourArea(c) for c in contours]
# x1= [cv2.contourArea(c) for c in s]
# print("z: \n ", z)
# print("x: \n ", x) 
# print("z: \n ", z1)
# print("x: \n ", x1) 
new_conts=Util.extract_cont_row(cont_sort_area)
Util.print_info(new_conts)
dr = cv2.drawContours(image,new_conts,-1,(000,000,255),2)
cv2.imshow("asdf",image)
cv2.waitKey()
pred=[]
for i in new_conts:
    (x,y,w,h) = cv2.boundingRect(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
    roi = thre[y:y+h,x:x+w]
    roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3)) 
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),block_norm="L2")
    # Util.print_info(roi_hog_fd,'true')
    # plt.plot(roi_hog_fd)
    # plt.show()
    nbr = model.predict(np.array([roi_hog_fd], np.float32))
    cv2.putText(image, str(int(nbr[0])), (x, y),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    cv2.namedWindow('image1111',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1111', 700,700)
    cv2.imshow("image1111",image)
    pred.append(nbr[0])
print(pred)
cv2.imwrite("test2.jpg",image)
cv2.waitKey()
cv2.destroyAllWindows()