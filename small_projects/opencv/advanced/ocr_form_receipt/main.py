import os
import cv2
import numpy as np
import pytesseract

# Initialize
per = 25
pixelThreshold = 500

roi = [[(98, 984), (680, 1074), 'text', 'Name'],
       [(740, 980), (1320, 1078), 'text', 'Phone'],
       [(98, 1154), (150, 1200), 'box', 'Sign'],
       [(738, 1152), (790, 1200), 'box', 'Allergic'],
       [(100, 1418), (686, 1518), 'text', 'Email'],
       [(740, 1416), (1318, 1512), 'text', 'ID'],
       [(110, 1598), (676, 1680), 'text', 'City'],
       [(748, 1592), (1328, 1686), 'text', 'Country']]

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/4.1.1'

imgQuery = cv2.imread('Query.png')
h, w, c = imgQuery.shape

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQuery, None)

path = 'UserForms'
myPicList = os.listdir(path)

for j, y in enumerate(myPicList):
    img = cv2.imread(path+"/"+y)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per/100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQuery, kp1, good[:100], None, flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    print(f'################## Extracting Data from Form {j}  ##################')

    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0,255,0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        if r[2] == 'text':
            print('{}:{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
            myData.append(pytesseract.image_to_string(imgCrop))

        if r[2] =='box':
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray,170,255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
        
            if totalPixels > pixelThreshold:
                totalPixels = 1
            else: 
                totalPixels = 0
            
            myData.append(totalPixels)
            print(f'{r[3]}:{totalPixels}')
            
        cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,255), 4)
    
    with open('DataOutput.cvs', 'a+') as f:
        for data in myData:
            f.write((str(data) + ','))
        
        f.write('\n')
    
    print(myData)
    cv2.imshow(y + "2", imgShow)
    cv2.imwrite(y, imgShow)

cv2.imshow("OUtput", imgQuery)
cv2.waitKey(0)