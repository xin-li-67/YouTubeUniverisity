import cv2
import utilities
import numpy as np

# Initialize params
webCamFeed = True
pathImage = "test.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 640
widthImg  = 480

utilities.initializeTrackbars()
count = 0

while True:
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    
    # real test img
    img = cv2.resize(img, (widthImg, heightImg))
    # create a blank image for tests if needed
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    thres = utilities.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Find all countours
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)

    # Find the biggest contour
    biggest, maxArea = utilities.biggestContour(contours)

    if biggest.size != 0:
        biggest = utilities.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0,255,0), 20)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # Remove 20 pixels from each side: avoid aliasing
        imgWarpColored = imgWarpColored[20 : imgWarpColored.shape[0] - 20, 20 : imgWarpColored[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # Apply adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BAYER_BG2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)
 
        imageArray = ([img, imgGray, imgThreshold, imgContours], [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours], [imgBlank, imgBlank, imgBlank, imgBlank])
 
    # Display labels
    lables = [["Original","Gray","Threshold","Contours"], ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]
 
    stackedImage = utilities.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Result", stackedImage)
 
    # Save image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50), (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1