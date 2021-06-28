import cv2
import numpy as np

# Use dlib to detect a face and find different facial landmarks on an image
import dlib

# Initialize
webcam = False
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
# Use the landmark data from dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def empty(a):
    pass

# Set up trackbars
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 640, 240)
cv2.createTrackbar("Blue", "BGR", 153, 255, empty)
cv2.createTrackbar("Green", "BGR", 0, 255, empty)
cv2.createTrackbar("Red", "BGR", 137, 255, empty)

# Create bounding box for specific part
def createBox(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255,255,255))
        img = cv2.bitwise_and(img, mask)
    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y : y+h, x : x+w]
        imgCrop = cv2.resize(imgCrop, (0,0), None, scale, scale)
        # cv2.imwrite("Mask.jpg", imgCrop)
        return imgCrop
    else:
        return mask

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread('test.jpg')
    
    # Resize the img
    img = cv2.resize(img, (0,0), None, 0.6, 0.6)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        # imgOriginal = cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = predictor(imgGray, face)
        # Loop all 68 landmarks to crop out all facial features into seperate images
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            # cv2.circle(imgOriginal, (x, y), 5, (50,50,255), cv2.FILLED)
            # cv2.putText(imgOriginal, str(n), (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1)

            ##########################################################################################
            # Jaw line = 0 to 16; Left Eyebrow = 17 to 21; Right Eyebrow = 22 to 26; Nose = 27 to 35 #
            # Left Eye = 36 to 41; Right Eye = 42 to 47; Lips = 48 to 60; Mouth Opening = 61 to 67   #
            ##########################################################################################

        # Identify every landmarks and mark them with idx numbers
        myPoints = np.array(myPoints)
        imgLips = createBox(img, myPoints[48:61], 3, masked=True, cropped=False)

        # Mask out lips and set it in purple color
        # maskLips = createBox(img, myPoints[48:61], masked=True, cropped=False)
        imgColorLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos("Blue", "BGR")
        g = cv2.getTrackbarPos("Green", "BGR")
        r = cv2.getTrackbarPos("Red", "BGR")
        imgColorLips[:] = b, g, r
        imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7,7), 10)
        imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
        imgColorLips = cv2.addWeighted(imgOriginalGray, 1, imgColorLips, 0.4, 0)
        cv2.imshow('BGR', imgColorLips)
        
    cv2.imshow("Original", imgOriginal)
    cv2.waitKey(1)