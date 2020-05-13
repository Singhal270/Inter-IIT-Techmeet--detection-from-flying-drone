import cv2 
import numpy as np 
import time
import math
import csv
import matplotlib.pyplot as plt
from matplotlib import style

############## setup plot frame

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

#########################################################
#    intial location list and function 
##################################################
xs=[]
ys=[]

def distance(x1,y1,x2,y2):
    dist=((x1-x2)**2 + (y1-y2)**2)**0.5
    return dist

#######################################################################
#    finding actual location of boxes
##############################################################
def location(img_x,img_y,gps_x,gps_y,box_x,box_y,ratio):
    
    x=gps_x + (box_x - img_x)*ratio
    y=gps_y + (box_y - img_y)*ratio

    return (x,y)
    
#######################################################
# find ratio
#             actual / 
#                   / pixel
###########################################



##################################################
#    live time plotting of cordinates of boxes
###########################################################






##########################################################
#  intial setup of some values 
#########################################################
lower_blue = np.array([40, 43, 0])	 
upper_blue = np.array([101, 255, 255])

kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])

kernelOpen=np.ones((5,5))
kernelClose=np.ones((11,11))



########################################################
#   capturing video frame by frame
#######################################################
vs = cv2.VideoCapture(0)
while(vs.isOpened()):
    ret, img = vs.read()
    start_time = time.time()
    
    cv2.imshow('before',img)

    
    d=img.shape     ####################
    y0 = int(d[0]/2)     # center of frame
    x0 = int(d[1]/2)     #####################



####################################################
#  finding final binary mask (condition of colour)
#####################################################
    edge = cv2.Canny(img,108,203)
    
    new = img.copy()    
    new[:,:,0]=edge     
    new[:,:,1]=edge
    new[:,:,2]=edge
    
    edge=cv2.addWeighted(img,0.63,new,0.37,0)
    edge = cv2.filter2D(edge, -1, kernel)
    
    hsv = cv2.cvtColor(edge, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernelOpen) 
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernelClose)
    #mask1 = cv2.GaussianBlur(mask1,(5,5),0)
    
    cv2.imshow('final mask',mask1)   ###### showing final mask

####################################################################
#  finding contours and center
##############################################################
    conts,h=cv2.findContours(mask1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    if len(conts)>0:
        for j in range (0,len(conts)):
                        if cv2.contourArea(conts[j])>500:   ##### condition of area 
                            print(cv2.contourArea(conts[j]))
                            x,y,w,h = cv2.boundingRect(conts[j])             ### making rectangle on contour
                            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  ########################################

                            approx = cv2.approxPolyDP(conts[j],0.01*cv2.arcLength(conts[j],True),True) ### condition of shapes
                            no_of_sides = len(approx)                                             ############################
                            
                            hull = cv2.convexHull(conts[j])                       
                            cv2.drawContours(img, [conts[j]], 0, (255, 0,0), 0)   ###drawing hull and contours
                            cv2.drawContours(img, [hull], 0,(0, 0, 255), 0)       ###############################

                            
                            M = cv2.moments(conts[j])                                      #### center of contours
                            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))  ########################
                            
###########################################################################################################
#                    call for actual loction by gps data (collect gps data )
###########################################################################################################





                            
###############################################################################################################################
#    make all continuous frame location to one
#############################################################################################
                            newcount=0    
                            for t in range(0,len(xs)):
                                if distance(center[1],center[0],xs[t],ys[t])< 100:
                                    xs[t]=(xs[t]+center[1])/2
                                    ys[t]=(ys[t]+center[0])/2
                                    newcount = newcount+1
                            if newcount==0:
                                xs.append(center[1])
                                ys.append(center[0])



###########################################################################################################
#      send location data     AND   call function for drawing graph of all location till this frame
###########################################################################################################
    




                                
################################################################
#   ending lines
#####################################################################

    
    #print(xs)     ##### x location list till this frame
    #print(ys)     ##### y location list till this frame

    
    cv2.imshow('after',img)
    #print("--- %s fps ---" % (1/(time.time() - start_time)))

    if cv2.waitKey(5) & 0xFF == 27:
        break
#print('final is = ' ,xs)   #### final location list of one drone
#print('final is = ',ys)    #######################################


plt.xticks(np.arange(0, 2*(x0), 100)) 
plt.yticks(np.arange(0, 2*y0, 100))     
ax1.clear()
ax1.scatter(xs,ys)
plt.show()



vs.release()
plt.release()
cv2.destroyAllwindow()












    
