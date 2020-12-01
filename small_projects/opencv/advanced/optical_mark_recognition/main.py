import cv2
import numpy as np
import utlis

# Initialize
webCapFeed = True
pathImage = "sample.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]

count = 0

while True:
    if webCapFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    
    img = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlue = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlue, 10, 70)

    try:
        # Find all contours
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)
        rectCon = utlis.getCornerPoints(contours)
        biggestPoints = utlis.getCornerPoints(rectCon[0])
        gradePoints = utlis.getCornerPoints(rectCon[1])

        if biggestPoints.size != 0 and gradePoints.size != 0:
            # Biggest rectangle warping
            biggestPoints = utlis.reorder(biggestPoints)
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0,255,0), 20)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0,0], [widthImg,0], [0,heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Second biggest rectangle
            cv2.drawContours(imgBigContour, gradePoints, -1, (255,0,0), 20)
            gradePoints = utlis.reorder(gradePoints)
            ptsG1 = np.float32(gradePoints)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            # Apply threshold
            imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpColored, 170, 255, cv2.THRESH_BINARY_INV)[1]

            # Get individual boxes
            boxes = utlis.splitBoxes(imgThresh)
            cv2.imshow("split Test", boxes[3])
            countR = 0
            countC = 0
            # Store the non zero values of each box
            myPixelVal = np.zeros((questions, choices))

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1

                if countC == choices:
                    countC = 0
                    countR += 1
            
            # Find the user answers and turn into a list
            myIndex = []

            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal)
            
            # Compare the value to find the correct ones
            grading = []

            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
                
            score = (sum(grading) / questions) * 100

            # Dispaly answers
            utlis.showAnswers(imgWarpColored, myIndex, grading, ans)
            utlis.drawGrid(imgWarpColored)
            # New blank image with the size of the warpped image
            imgRawDrawings = np.zeros_like(imgWarpColored)
            utlis.showAnswers(imgRawDrawings, myIndex, grading, ans)
            # Inverse the transformation matrix
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            # Display grade
            imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)
            cv2.putText(imgRawGrade, str(int(score))+"%", (70, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,255), 3)
            invGradeMatrix = cv2.getPerspectiveTransform(ptsG2, ptsG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invGradeMatrix, (widthImg, heightImg))

            # Show answer and grade on final image
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            imgArray = ([img, imgGray, imgCanny, imgContours], [imgBigContour, imgThresh, imgWarpColored, imgFinal])
            cv2.imshow("Final Result", imgFinal)
    except:
        imgArray = ([img, imgGray, imgCanny, imgContours], [imgBlank, imgBlank, imgBlank, imgBlank])
    
    lables = [["Original", "Canny", "Edges", "Contours"], ["Biggest Contour", "Threshold", "Warpped", "Final"]]

    stackedImage = utlis.stackImages(imgArray, 0.5, lables)
    cv2.imshow("Result", stackedImage)

    # Save images
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1]/2) - 230), int(stackedImage.shape[0]/2) + 50), (1100, 350), (0,255,0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1]/2) - 200, int(stackedImage.shape[0]/2)), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1