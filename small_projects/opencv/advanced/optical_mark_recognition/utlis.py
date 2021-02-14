import cv2
import numpy as np

def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows
        hor_con = [imgBlank] * cols

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cvt.COLOR_GRAY2BGR)
        
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)

        for x in range(0, rows):
            for y in range(0, cols):
                cv2.rectangle(ver, (y*eachImgWidth, x*eachImgHeight), (y*eachImgWidth+len(lables[x])*13+27, 30+eachImgHeight*x), (255,255,255), cv2.FILLED)
                cv2.putText(ver, lables[x], (y*eachImgWidth+10, x*eachImgHeight+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,255), 2)
    
    return ver

def reorder(myPoints):
    # Remove extra brackets
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]   #[0,0]
    myPointsNew[3] = myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  #[h,0]

    return myPointsNew

# Find only rectangle contours
def rectContour(contours):
    rectCon = []
    area = 0

    for i in contours:
        area = cv2.contourArea(i)
        
        if area > 50:
            peri = cv2.arcLength(i, True)                    # perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # current contour, resolution, closed

            if len(approx) == 4:
                rectCon.append(i)
    
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)

    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)

    return approx

def splitBoxes(img):
    rows = np.vsplit(img, 5)
    boxes = []

    for r in rows:
        cols = np.hsplit(r, 5)
        
        for box in cols:
            boxes.append(box)

    return boxes

def drawGrid(img, questions=5, choices=5):
    secW = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)
    
    for i in range (0,9):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        pt3 = (secW * i, 0)
        pt4 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255,255,0), 2)
        cv2.line(img, pt3, pt4, (255,255,0), 2)

    return img

def showAnswers(img, myIndex, grading, ans, questions=5, choices=5):
    secW = int(img.shape[1] / questions)
    secH = int(img.shape[0] / choices)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2

        if grading[x]==1:
            myColor = (0,255,0)
            #cv2.rectangle(img, (myAns*secW, x*secH), ((myAns*secW)+secW, (x*secH)+secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
        else:
            myColor = (0,0,255)
            #cv2.rectangle(img, (myAns*secW, x*secH), ((myAns*secW)+secW, (x*secH)+secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # Correct ones
            myColor = (0,255,0)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns*secW)+secW//2, (x*secH)+secH//2), 20, myColor, cv2.FILLED)