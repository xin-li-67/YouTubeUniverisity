# for collecting images and log in a seperate folder for next step training
import os
import cv2
import pandas as pd
from datetime import datetime

global imgList, steeringList
countFolder = 0
count = 0
imgList = []
steeringList = []

myDirectory = os.path.join(os.getcwd(), 'materials')

# CREATE A NEW FOLDER BASED ON THE PREVIOUS FOLDER COUNT
while os.path.exists(os.path.join(myDirectory, f'IMG{str(countFolder)}')):
        countFolder += 1

newPath = myDirectory +"/IMG"+str(countFolder)
os.makedirs(newPath)

def saveData(img, steering):
    global imgList, steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.', '')
    filename = os.path.join(newPath, f'Image_{timestamp}.jpg')
    cv2.imwrite(filename, img)
    imgList.append(filename)
    steeringList.append(steering)

def saveLog():
    global imgList, steeringList
    rawData = {'Image': imgList, 'Steering': steeringList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory, f'log_{str(countFolder)}.csv'), index=False, header=False)

    print('Log Saved')
    print('Total Images: ', len(imgList))

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    for x in range(10):
        _, img = cap.read()
        saveData(img, 0.5)
        cv2.waitKey(1)
        cv2.imshow("Image", img)
    saveLog()