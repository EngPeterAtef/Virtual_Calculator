import numpy as np
import cv2
import imutils

bg = None
top, right, bottom, left = 10, 350, 225, 590

cap = cv2.VideoCapture(0)

def weightedaverage(image):
    global bg, top, right, bottom, left
    print("Minor 3-4 Seconds lag")
    bg = image.astype("float")

    for i in range (0,30):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[top:bottom, right:left]

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi,(17,17),0)

        cv2.accumulateWeighted(roi, bg, 0.5)

    return

def initialize():
    global cap, bg, top, right, bottom, left
    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[top:bottom, right:left]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi,(17,17),0)
        roi_copy = roi.copy()
        et,thresh1 = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('fframe', frame)
        cv2.imshow('fthresh', thresh1)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            weightedaverage(roi_copy)
            cv2.destroyAllWindows()
            return
            


print("Initializing Frame, Keep Camera Still, Wait for few Seconds")
initialize()
print("Done boi, now wave your hand")

while (True):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[top:bottom, right:left]
    roi_copy = roi.copy()

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi,(17,17),0)

    roi = cv2.absdiff(bg.astype('uint8'), roi)
    #roi = cv2.bitwise_not(roi)
    
    et,thresh1 = cv2.threshold(roi,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)

    '''
    hull = cv2.convexHull(max_cnt, returnPoints = True)
    hull=hull.reshape(hull.shape[0],hull.shape[2])
    cv2.polylines(roi_copy, [hull],True,(255,255,0),3)
    '''

    '''
    M = cv2.moments(max_cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.drawContours(roi_copy, contours, -1, (0, 0, 255), 2)
    cv2.circle(roi_copy, (cx, cy), 7, (0, 0, 0), -1)
    '''

    (x,y),radius = cv2.minEnclosingCircle(max_cnt)
    center = (int(x),int(y))
    cv2.circle(roi_copy, center, 7, (255, 0, 0), -1)

    hull = cv2.convexHull(max_cnt, returnPoints = False)
    defects = cv2.convexityDefects(max_cnt, hull)
    
    
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(max_cnt[s][0])
        end = tuple(max_cnt[e][0])
        far = tuple(max_cnt[f][0])
        cv2.line(roi_copy,start,end,[0,255,0],2)
        cv2.circle(roi_copy,far,5,[0,0,255],-1)
    
    
    #cv2.drawContours(roi_copy, max_cnt, -1 , (0, 0, 255), 2)
    #cv2.drawContours(roi_copy, hull, -1 , (0, 0, 255), 5)
    

    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('threshold', thresh1)
    cv2.imshow('contours', roi_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #print(range(defects.shape[0]))
        break

cap.release()
cv2.destroyAllWindows()