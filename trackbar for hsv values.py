import cv2 
import numpy as np 


def nothing(x):
    print(x)
cv2.namedWindow('img')
cv2.resizeWindow('img',600,800)
cv2.createTrackbar('min_h','img',0,255,nothing)
cv2.createTrackbar('min_s','img',0,255,nothing)
cv2.createTrackbar('min_v','img',0,255,nothing)
cv2.createTrackbar('max_h','img',0,255,nothing)
cv2.createTrackbar('max_s','img',0,255,nothing)
cv2.createTrackbar('max_v','img',0,255,nothing)
cv2.createTrackbar('kernelOpen','img',0,100,nothing)
cv2.createTrackbar('kernelClose','img',0,100,nothing)
cv2.createTrackbar('Gaussian','img',0,100,nothing)
cv2.createTrackbar('cannyedge_lower','img',0,1000,nothing)
cv2.createTrackbar('cannyedge_upper','img',0,1000,nothing)
cv2.createTrackbar('edge_img_add_weight','img',0,100,nothing)






vs = cv2.VideoCapture(0)
while(vs.isOpened()):
    ret, img = vs.read()

    min_h = cv2.getTrackbarPos('min_h','img')
    min_s = cv2.getTrackbarPos('min_s','img')
    min_v = cv2.getTrackbarPos('min_v','img')
    max_h = cv2.getTrackbarPos('max_h','img')
    max_s = cv2.getTrackbarPos('max_s','img')
    max_v = cv2.getTrackbarPos('max_v','img')
    kernelOpen = cv2.getTrackbarPos('kernelOpen','img')
    kernelClose = cv2.getTrackbarPos('kernelClose','img')
    cannyedge_lower = cv2.getTrackbarPos('cannyedge_lower','img')
    cannyedge_upper = cv2.getTrackbarPos('cannyedge_upper','img')
    edge_img_add_weight = cv2.getTrackbarPos('edge_img_add_weight','img')
    Gaussian = cv2.getTrackbarPos('Gaussian','img')

    lower_blue = np.array([min_h,min_s,min_v])	 
    upper_blue = np.array([max_h,max_s,max_v])

    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])

    kernelOpen=np.ones((kernelOpen,kernelOpen))
    kernelClose=np.ones((kernelClose,kernelClose))

    edge = cv2.Canny(img,cannyedge_lower,cannyedge_upper)
    
    new = img.copy()    
    new[:,:,0]=edge     
    new[:,:,1]=edge
    new[:,:,2]=edge
    
    edge=cv2.addWeighted(img,1-(edge_img_add_weight)/100,new,(edge_img_add_weight)/100,0)
    edge = cv2.filter2D(edge, -1, kernel)
    
    hsv = cv2.cvtColor(edge, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernelOpen) 
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernelClose)
    mask1 = cv2.GaussianBlur(mask1,(5,5),0)
    
    cv2.imshow('final mask',mask1)   ###### showing final mask

    conts,h=cv2.findContours(mask1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for j in range (0,len(conts)):
        
        x,y,w,h = cv2.boundingRect(conts[j])             ### making rectangle on contour
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  ########################################

        hull = cv2.convexHull(conts[j])                       
        cv2.drawContours(img, [conts[j]], 0, (255, 0,0), 0)   ###drawing hull and contours
        cv2.drawContours(img, [hull], 0,(0, 0, 255), 0) 


    cv2.imshow('after',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break



vs.release()
cv2.destroyAllwindow()

