import numpy as np
import cv2
import Util
#contours=[(212, 169, 1, 1), (220, 163, 16, 7), (49, 103, 28, 65), (93, 101, 27, 65), (136, 100, 27, 65), (179, 97, 28, 66), (0, 65, 21, 143), (117, 55, 16, 10), (48, 29, 27, 64), (83, 28, 20, 63), (228, 27, 1, 3), (143, 25, 26, 64), (175, 24, 28, 64), (23, 23, 2, 1), (223, 17, 5, 4), (16, 0, 253, 208)]

image = cv2.imread("test.jpg")
im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
_,contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
def serch_number_bouding_rect(contours):
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=lambda x: (x[3]*x[2]), reverse=True)
    #print(contours[1][2])
    print("Contour: ",rects)
    print(np.shape(rects))
    temp_rects = rects.copy()
    for i,rect in enumerate(rects):
        #print("i",i)
        #print(str(rect[3]/rect[2]) + "   " + str((rect[3]*rect[2])) + " ratio: " + str((rects[0][2]*rects[0][3])/(rect[3]*rect[2])))
        print ("Cont: " + str(rect) + " " + str(rect[3]/rect[2] <= 2) + " " + str(rect[3]/rect[2] >= 3.5) + " " + str((rects[0][2]*rects[0][3])/(rect[3]*rect[2]) <= 7) + " " + str((rects[0][2]*rects[0][3])/(rect[3]*rect[2]) >=45 ))
        if rect[3]/rect[2] <= 2 or rect[3]/rect[2] >= 3.8 or (rects[0][2]*rects[0][3])/(rect[3]*rect[2]) <= 7 or (rects[0][2]*rects[0][3])/(rect[3]*rect[2]) >=45:
            temp_rects.remove(rect)
            print("remove: " + str(rect) + " area: " + str((rect[3]*rect[2]))+ " ratio: " + str((rects[0][2]*rects[0][3])/(rect[3]*rect[2])))
    print("Bouding rect: ", temp_rects)
    return temp_rects

def sort_boxes_to_2_row(BoundingBoxes):
    Boxes = sorted(BoundingBoxes,key=lambda x:x[1])
    anchor = Boxes[0][1] + Boxes[0][3]
    h1=[]
    h2=[]
    for box in Boxes:
        if box[1] < anchor:
            h1.append(box)
        else:
            h2.append(box)
    h1_sorted = sorted(h1,key=lambda x:x[0],reverse=False)
    h2_sorted = sorted(h2,key=lambda x:x[0],reverse=False)
    print(h1_sorted)
    print(h2_sorted)
    return (h1_sorted+h2_sorted)

def Draw_BoudingBoxes(image,BoundingBoxes):
    image = cv2.imread(image)
    #Boxes = sort_boxes(Boxes)
    for box in BoundingBoxes:
        (x,y,w,h) = box
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.imshow("ASDF",image)
        cv2.waitKey()
    
Boxes = serch_number_bouding_rect(contours)
Boxes=sort_boxes_to_2_row(Boxes)
Draw_BoudingBoxes("test.jpg",[(255, 79, 29, 103)])
print("BOXES: ",Boxes)
#print(serch_number_bouding_rect(contours))
