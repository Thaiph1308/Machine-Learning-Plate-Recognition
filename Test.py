import numpy as np
contours=[(212, 169, 1, 1), (220, 163, 16, 7), (49, 103, 28, 65), (93, 101, 27, 65), (136, 100, 27, 65), (179, 97, 28, 66), (0, 65, 21, 143), (117, 55, 16, 10), (48, 29, 27, 64), (83, 28, 20, 63), (228, 27, 1, 3), (143, 25, 26, 64), (175, 24, 28, 64), (23, 23, 2, 1), (223, 17, 5, 4), (16, 0, 253, 208)]
def serch_number_bouding_rect(contours):
    #rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(contours,key=lambda x: (x[3]*x[2]), reverse=True)
    #print(contours[1][2])
    print("Contour: ",rects)
    print(np.shape(rects))
    temp_rects = rects.copy()
    for i,rect in enumerate(rects):
        #print("i",i)
        #print(str(rect[3]/rect[2]) + "   " + str((rect[3]*rect[2])) + " ratio: " + str((rects[0][2]*rects[0][3])/(rect[3]*rect[2])))
        print ("Cont: " + str(rect) + " " + str(rect[3]/rect[2] <= 2) + " " + str(rect[3]/rect[2] >= 3.5) + " " + str((contours[0][2]*contours[0][3])/(rect[3]*rect[2]) <= 7) + " " + str((contours[0][2]*contours[0][3])/(rect[3]*rect[2]) >=45 ))
        if rect[3]/rect[2] <= 2 or rect[3]/rect[2] >= 3.5 or (rects[0][2]*rects[0][3])/(rect[3]*rect[2]) <= 7 or (rects[0][2]*rects[0][3])/(rect[3]*rect[2]) >=45:
            temp_rects.remove(rect)
            print("remove: " + str(rect) + " area: " + str((rect[3]*rect[2])))
    print("Bouding rect: ", temp_rects)
    return temp_rects

#print(serch_number_bouding_rect(contours))
