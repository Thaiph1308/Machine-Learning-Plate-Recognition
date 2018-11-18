import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Util
import pickle
with open('SVM90.pkl', 'rb') as fid:
    model = pickle.load(fid)
#np.set_printoptions(threshold=np.nan)

#Load image và convert sang image gray
im = cv2.imread("bxx3.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# lọc nhiễu bằng bilateralFilter mục đích lọc này là làm tăng strengt cho edge trên image
noise_removal = cv2.bilateralFilter(im_gray,9,75,75)
# Cân bằng lại histogram của ảnh 
equal_histogram = cv2.equalizeHist(noise_removal)
#  Morphogoly open mục đích là làm tăng dilation của edge và giảm edge nhiễu
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel1,iterations=20)
# subtract image
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
# dùng threshold OSTU
ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
# Dùng canny
canny_image = cv2.Canny(thresh_image,250,255)
# dilation
kernel2= np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel2)
#
new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(np.size(contours))

contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]

for c in contours:
    # print("New contour")
    # print (c)
    # print("\n")
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True) 
    if len(approx) == 4:
            screenCnt = approx
            break
#final = cv2.drawContours(im, [screenCnt], -1, (0, 255, 0), 3)

###
# plt.subplots(figsize=(20,20))
# plt.subplot(1,4,1)
# plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
# plt.title("image")
# plt.axis('off')
# plt.subplot(1,4,2)
# plt.imshow(im_gray,cmap="gray")
# plt.title("im_gray")
# plt.axis('off')
# plt.subplot(1,4,3)
# plt.imshow(noise_removal,cmap="gray")
# plt.title("noise_removal")
# plt.axis('off')
# plt.subplot(1,4,4)
# plt.imshow(morph_image,cmap="gray")
# plt.title("morph_image")
# plt.axis('off')
# plt.subplots(figsize=(20,20))
# plt.subplot(1,4,1)
# plt.imshow(sub_morp_image,cmap="gray")
# plt.title("sub_morp_image")
# plt.axis('off')
# plt.subplot(1,4,2)
# plt.imshow(thresh_image,cmap="gray")
# plt.title("thresh_image")
# plt.axis('off')
# plt.subplot(1,4,3)
# plt.imshow(canny_image,cmap="gray")
# plt.title("canny_image")
# plt.axis('off')
# plt.subplot(1,4,4)
# plt.imshow(dilated_image,cmap="gray")
# plt.title("dilated_image")
# plt.axis('off')
###
(x,y,w,h) = cv2.boundingRect(screenCnt)
roi = im[y:y+h+30,x:x+w+30]
Util.print_info(roi)
# cv2.imshow("roi",roi)
# cv2.waitKey()
###Extract Bien so

biensoxe = roi.copy()

################
cv2.imwrite("test.jpg",biensoxe)
roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
roi_blur = cv2.GaussianBlur(roi_gray,(3,3),1)
ret,thre = cv2.threshold(roi_blur,120,255,cv2.THRESH_BINARY_INV)
#cany = cv2.Canny(thre,250,255)
kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
thre_mor = cv2.morphologyEx(thre,cv2.MORPH_DILATE,kerel3)
##
# cv2.imshow("im1",roi)
# cv2.imshow("im3",thre_mor)
# cv2.imshow("im2",roi1)
# cv2.waitKey()
# cv2.destroyAllWindows()
##
_,cont,hier = cv2.findContours(thre_mor,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#dr = cv2.drawContours(biensoxe,cont,-1,(0,255,0))
# cv2.imshow("roi1",roi1) ## Split Plate + contour all 
# cv2.waitKey()
#
print("cont: ",len(cont))
print(type(cont))
print(np.shape(cont))
print(cont[0])
# cv2.drawContours(biensoxe,cont,-1,(255,0,0),1)
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600,600)
# cv2.imshow("image",biensoxe)
# cv2.waitKey()

areas_ind = {}
areas = []
boundingRect = []

for ind,cnt in enumerate(cont):
    area = cv2.contourArea(cnt)
    # x= cv2.boundingRect(cnt)
    # #print_info(x,'true')
    # boundingRect.append(x)
    areas_ind[ind] = area
    areas.append(area)
# boundingRect=np.asarray(boundingRect)
# Util.print_info(boundingRect,'true')
# #print("Bdr: ", boundingRect)
# print(boundingRect[0])
# print(boundingRect[0][1])

print(len(areas_ind))
print("areas_ind: ", areas_ind)
areas = sorted(areas,reverse=True)
print("areas roi: ", areas)
areas = sorted(areas,reverse=True)[1:11]
print("areas: ", areas)
# #print(areas_ind[areas])
# cnt = sorted(areas_ind,key=lambda key: areas,reverse=True)
# print("cnt: ", cnt )
# print("Areas_ind[1]:", areas_ind[51])

new_cont=[]
for i,area in enumerate(areas):
    #print(list(areas_ind.keys())[list(areas_ind.values()).index(area)],"index: ", i , "area: ", area)  
    new_cont.append(cont[list(areas_ind.keys())[list(areas_ind.values()).index(area)]])
    (x,y,w,h) = cv2.boundingRect(cont[list(areas_ind.keys())[list(areas_ind.values()).index(area)]])
    #cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),1)
#cv2.imshow("ewqr",roi)
#cv2.waitKey()

Util.print_info(new_cont)
(conts, boundingBox)=Util.sort_contours(new_cont)
Util.print_info(boundingBox,'true')
# for (i,c) in enumerate(conts):
#     Util.draw_contour(roi,c,i)
character = []
for box in boundingBox:
    (x,y,w,h) = box
    image = roi[y:y+h,x:x+w]
    character.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
#Util.sub_plot(character,1,10)
Util.print_info(character[2],'true')
cv2.imwrite("testsvm.jpg",biensoxe)
#pred=model.predict(character[1])
#print("PREDICT: ", pred)
# Util.draw_contour(roi,conts[1],0)
# (x,y,w,h) = boundingBox[1]
# cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),1)
#cv2.imshow("Sorted", roi)
#cv2.waitKey(0)
# character = []
# for area in areas:
#     (x,y,w,h) = cv2.boundingRect(cont[list(areas_ind.keys())[list(areas_ind.values()).index(area)]])
#     image = roi[y:y+h,x:x+w]
#     character.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

#print(cont[list(areas_ind.keys())[list(areas_ind.values()).index(2)]])
#cv2.drawContours(roi,cont[list(areas_ind.keys())[list(areas_ind.values()).index(2)]],-1,(0,255,0),1)
#print(new_cont)
#print(new_cont)
# print(type(new_cont))
# print(len(new_cont))
# print(np.shape(new_cont))
# plt.imshow(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
# cv2.namedWindow('image1111',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image1111', 700,700)
# cv2.imshow("image1111",biensoxe)
#dr = cv2.drawContours(roi,new_cont,-1,(0,255,0))
# cv2.imshow("image1111",roi)
# cv2.waitKey()
# sub_plot(character,1,10)
