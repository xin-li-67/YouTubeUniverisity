import os
import solver

from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
image = "resources/2.jpg"
height = 450
width = 450
model = initialize_model()
########################################################################

img = cv2.imread(image)
img = cv2.resize(img, (width, height))
imgBlnk = np.zeros((height, width, 3), np.uint8)
imgThre = preprocessing(img)
imgContours = img.copy()
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours, contours, -1, (0,255,0), 3)

biggest, maxArea = biggest_contour(contours)
print(biggest)

if biggest.size != 0:
    biggest = reorder(biggest)
    print(biggest)

    cv2.drawContours(imgBigContour, biggest, -1, (0,0,255), 25)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))
    imgDetectedDigits = imgBlnk.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    imgSolvedDigits = imgBlnk.copy()
    boxes = split_boxes(imgWarpColored)
    print(len(boxes))
    
    numbers = get_predictions(boxes, model)
    print(numbers)
    
    imgDetectedDigits = display_number(imgDetectedDigits, numbers, color=(255,0,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)

    board = np.array_split(numbers,9)
    print(board)

    try:
        solver.solve(board)
    except:
        pass
    print(board)

    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers =flatList * posArray
    imgSolvedDigits= display_number(imgSolvedDigits, solvedNumbers)

    pts2 = np.float32(biggest)
    pts1 =  np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (width, height))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = draw_grid(imgDetectedDigits)
    imgSolvedDigits = draw_grid(imgSolvedDigits)

    imageArray = ([img, imgThre, imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
    stackedImage = stack_images(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)
else:
    print("Nothing")

cv2.waitKey(0)