import cv2 
import numpy as np


vs = cv2.VideoCapture(0)
while(vs.isOpened()):
    ret, img = vs.read()
    
    d=img.shape          ####################
    y0 = int(d[0]/2)     # center of frame
    x0 = int(d[1]/2)     #####################

    print(d)


    cv2.imshow('after',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllwindow()
