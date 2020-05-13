import cv2 
import numpy as np
import math

actual_distance = 80  #### 80cm is actual 

lower_blue = np.array([40, 40, 40])	 
upper_blue = np.array([100, 255, 255])

kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])

kernelOpen=np.ones((8,8))
kernelClose=np.ones((10,10))





vs = cv2.VideoCapture(1)
while(vs.isOpened()):
    ret, img = vs.read()
    
    d=img.shape          ####################
    y0 = int(d[0]/2)     # center of frame
    x0 = int(d[1]/2)     #####################

    print(d)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernelOpen) 
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernelClose)
    mask1 = cv2.GaussianBlur(mask1,(5,5),0)
    conts,h=cv2.findContours(mask1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    if len(conts)>0:
        x1,y1,w1,h1 = cv2.boundingRect(conts[0])
        cx1 = int(x1+(w1/2))
        cy1 = int(y1+(h1/2))
        cv2.circle(img,(cx1,cy1), 2, (255,0,0),2)

        
        x2,y2,w2,h2 = cv2.boundingRect(conts[1])
        cx2 = int(x1+(w1/2))
        cy2 = int(y1+(h1/2)
                  
        cv2.circle(img,(cx2,cy2), 2, (255,0,0),2)
                  
        distance = ( (cx1 - cx2)**2 + (cy1 - cy2)**2 )**0.5
        print(distance)
        print("ratio is = " ,actual_distance/distance)
        cv2.line(img, (cx1, cy1), (cx2, cy2), (0,255,0),1)
    

    cv2.imshow('after',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllwindow()
