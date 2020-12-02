import cv2
import numpy as np

def detect_color(img, hsv):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv[0], hsv[2], hsv[4]])
    upper = np.array([hsv[1], hsv[3], hsv[5]])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    return imgResult

def get_contours(img, imgDraw, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgDraw = imgDraw.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.array((10,10))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgClose = cv2.morphologyEx(imgDial, cv2.MORPH_CLOSE, kernel)

    if showCanny: 
        cv2.imshow('Canny', imgClose)
    contours, hiearchy = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])

    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            x, y, w, h = con[3]
            cv2.rectangle(imgDraw, (x, y), (x + w, y + h), (255,0,255), 3)
    
    return imgDraw, finalCountours

def get_roi(img, contours):
    roiList = []
    for con in contours:
        x, y, w, h = con[3]
        roiList.append(img[y:y + h, x:x + w])
    
    return roiList


def roi_display(roiList):
    for x, roi in enumerate(roiList):
        roi = cv2.resize(roi, (0,0), None, 2, 2)
        cv2.imshow(str(x), roi)


def save_text(highlightedText):
    with open('detection.csv', 'w') as f:
        for text in highlightedText:
            f.writelines(f'\n{text}')


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
        
        imgBlnk = np.zeros((height,width,3), np.uint8)
        hor = [imgBlnk] * rows
        hor_con = [imgBlnk] * rows

        for i in range(0, rows):
            hor[i] = np.hstack(imgArray[i])
        ver = np.vstack(hor)
    else:
        for i in range(0, rows):
            if imgArray[i].shape[:2] == imgArray[0].shape[:2]:
                imgArray[i] = cv2.resize(imgArray[i], (0, 0), None, scale, scale)
            else:
                imgArray[i] = cv2.resize(imgArray[i], (imgArray[0].shape[1],imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[i].shape) == 2:
                imgArray[i] = cv2.cvtColor(imgArray[i], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver