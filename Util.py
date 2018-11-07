import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def extract_cont_row(contours):
	h1 =[]
	h2 =[]
	(cnts, boundingBoxes) = sort_contours(contours)
	anchor = boundingBoxes[0][1]+ boundingBoxes[0][3]
	for i,box in enumerate(boundingBoxes):
		if box[1] <= anchor:
			h1.append(cnts[i])
		else:
			h2.append(cnts[i])
	return h1+h2