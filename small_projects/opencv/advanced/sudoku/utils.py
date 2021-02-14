import cv2
import numpy as np
from tensorflow.keras.models import load_model

def initialize_model():
    model = load_model('resources/model.h5')
    return model

def preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgThre = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)

    return imgThre

# reorder points for warpping
def reorder(points):
    points = points.reshape((4,2))
    new_points = np.zeros((4,1,2), dtype=np.int32)
    
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0

    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    
    return biggest, max_area

# split the img into 81 different parts
def split_boxes(img):
    rows = np.vsplit(img, 9)
    boxes = []

    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    
    return boxes

def get_predictions(boxes, model):
    res = []
    for image in boxes:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1]  -4]
        img = cv2.resize(img, (28,28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        
        if probabilityValue > 0.8:
            res.append(classIndex[0])
        else:
            res.append(0)

    return res

def display_number(img, numbers, color=(0,255,0)):
    w = int(img.shape[1] / 9)
    h = int(img.shape[0] / 9)

    for i in range(0, 9):
        for j in range(0, 9):
            if numbers[(9 * j) + i] != 0:
                cv2.putText(img, str(numbers[(9 * j) + i]), (i * w + int(w / 2) - 10, int((j + 0.8) * h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)

    return img

def draw_grid(img):
    w = int(img.shape[1] / 9)
    h = int(img.shape[0] / 9)

    for i in range(0, 9):
        p1 = (0, h * i)
        p2 = (img.shape[1], h * i)
        p3 = (w * i, 0)
        p4 = (w * i, img.shape[0])

        cv2.line(img, p1, p2, (255,255,0), 2)
        cv2.line(img, p3, p4, (255,255,0), 2)

    return img

def stack_images(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rows_available = isinstance(imgArray[0], list)
    w = imgArray[0][0].shape[1]
    h = imgArray[0][0].shape[0]

    if rows_available:
        for i in range(0, rows):
            for j in range(0, cols):
                imgArray[i][j] = cv2.resize(imgArray[i][j], (0,0), None, scale, scale)
                if len(imgArray[i][j].shape) == 2:
                    imgArray[i][j] = cv2.cvtColor(imgArray[i][j], cv2.COLOR_GRAY2BGR)
        
        img_blank = np.zeros((h, w, 3), np.uint8)
        hor = [img_blank] * rows
        hor_con = [img_blank] * rows

        for i in range(0, rows):
            hor[i] = np.hstack(imgArray[i])
            hor_con[i] = np.concatenate(imgArray[i])
        
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for i in range(0, rows):
            imgArray[i] = cv2.resize(imgArray[i], (0,0), None, scale, scale)
            if len(imgArray[i].shape) == 2:
                imgArray[i] = cv2.cvtColor(imgArray[i], cv2.COLOR_GRAY2BGR)
        
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor

    return ver