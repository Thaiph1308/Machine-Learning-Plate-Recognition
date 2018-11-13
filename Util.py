import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

def sub_plot(characters,row,column):
    plt.subplots(figsize=(10,10))
    for i,character in enumerate(characters):
        plt.subplot(row,column,i+1)
        plt.imshow(character)
    plt.show()
    
def print_info(x,print_full='false'):
    print("np.shape(x): ",np.shape(x))
    print("type(x): ", type(x))
    print("len(x): ", len(x))
    if print_full =='true':
        print(x)
        return True
    else: 
        return False
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
	
def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
 
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)

	# return the image with the contour number drawn on it
	return image

def extract_cont_row(contours,return_bounding_boxes=False):
	h1 =[]
	h2 =[]
	(cnts, boundingBoxes) = sort_contours(contours,method='top-to-bottom')
	anchor = boundingBoxes[0][1]+ boundingBoxes[0][3]
	for i,box in enumerate(boundingBoxes):
		if box[1] <= anchor:
			h1.append(cnts[i])
		else:
			h2.append(cnts[i])
	(h1,h1_boudingboxes) = sort_contours(h1)
	(h2,h2_boudingboxes) = sort_contours(h2)
	if return_bounding_boxes == False:
		return (h1+h2)
	else:
		return (h1+h2,h1_boudingboxes+h2_boudingboxes)

def image_reshape(image,img_height=28,img_width=28):
	im=cv2.imread(image)
	im=cv2.resize(im,(img_height,img_width))
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	print(np.shape(im))
	arr = np.reshape(im,(img_height,img_width))
	print(np.shape(arr))
	arr = np.expand_dims(arr,axis=0)
	return arr

def image_reshape_2(image,img_height=28,img_width=28):
	im=cv2.resize(image,(img_height,img_width))
	# cv2.imshow("asdf",im)
	# cv2.waitKey()
	arr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("asdf",arr)
	# cv2.waitKey()
	#print("Np shape grayscale: ",np.shape(arr))
	#arr = np.reshape(im,(img_height,img_width))
	#print("NP shape after np reshape",np.shape(arr))
	#print(arr)
	arr = img_to_array(arr)
	# plt.imshow(arr.squeeze(),aspect="auto")
	# plt.show()
	#print("NP shape after img to array",np.shape(arr))
	#arr = np.expand_dims(arr,axis=0)
	#print("NP shape after expand_dims",np.shape(arr))
	#print(arr)
	return arr