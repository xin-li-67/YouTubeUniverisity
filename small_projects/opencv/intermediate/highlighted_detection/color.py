import cv2
import numpy as np

def empty(a):
    pass

def stack_images(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rows_available = isinstance(imgArray[0], list)
    w = imgArray[0][0].shape[1]
    h = imgArray[0][0].shape[0]

    if rows_available:
        for i in range(0, rows):
            for j in range(0, cols):
                if imgArray[i][j].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[i][j] = cv2.resize(imgArray[i][j], (0,0), None, scale, scale)
                else:
                    imgArray[i][j] = cv2.resize(imgArray[i][j], (imgArray[0][0].shape[1],imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[i][j].shape) == 2:
                    imgArray[i][j] = cv2.cvtColor(imgArray[i][j], cv2.COLOR_GRAY2BGR)
        
        imgBlnk = np.zeros((h, w, 3), np.uint8)
        hor = [imgBlnk] * rows
        hor_con = [imgBlnk] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        
        ver = np.vstack(hor)
    else:
        for i in range(0, rows):
            if imgArray[i].shape[:2] == imgArray[0].shape[:2]:
                imgArray[i] = cv2.resize(imgArray[i], (0,0), None, scale, scale)
            else:
                imgArray[i] = cv2.resize(imgArray[i], (imgArray[0].shape[1],imgArray[0].shape[0]), None, scale, scale)

            if len(imgArray[i].shape) == 2:
                imgArray[i] = cv2.cvtColor(imgArray[i], cv2.COLOR_GRAY2BGR)
        
        hor = np.hstack(imgArray)
        ver = hor
    
    return ver

path = 'test.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars",0 , 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    imgStack = stack_images(0.6, ([img,imgHSV], [mask,imgResult]))
    cv2.imshow("Stacked Images", imgStack)

    cv2.waitKey(1)