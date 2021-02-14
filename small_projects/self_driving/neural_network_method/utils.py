import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# initialize
def getImage(filePath):
    myImagePathL = filePath.split('/')[-2:]
    myImagePath = os.path.join(myImagePathL[0], myImagePathL[1])
    return myImagePath

def importDataInfo(path):
    cols = ['Center', 'Steering']
    folderNo = len(os.listdir(path)) // 2
    data = pd.DataFrame()

    for x in range(17, 22):
        newData = pd.read_csv(os.path.join(path, f'log_{x}.cvs'), names=cols)
        newData['Center'] = newData['Center'].apply(getImage)
        data = data.append(newData, True)
    
    print(' ')
    print('Total Number of Images', data.shape(0))
    return data

# visualize and balance the data
def dataBalancer(data, display=True):
    binNo = 31
    samplesPerBin = 300
    hist, bins = np.histogram(data['Steering'], binNo)

    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Data Visualisation')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    
    removeIndexList = []

    for i in range(binNo):
        binDataList = []
        for j in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    
    print('Removed Images:', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print('Remaining Images:', len(data))
    
    if display:
        hist, _ = np.histogram(data['Steering'], (binNo))
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()
    
    return data

# prepare
def loadData(path, data):
    imgPath = []
    steering = []

    for i in range(len(data)):
        indexdData = data.iloc[i]
        imgPath.append(os.path.join(path, indexdData[0]))
        steering.append(float(indexdData[1]))
    
    imgPath = np.asarray(imgPath)
    steering = np.asarray(steering)

    return imgPath, steering

# augment
def augmentImage(imgPath, steering):
    img =  mpimg.imread(imgPath)
    
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    
    return img, steering

# preprocess
def preprocessing(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img / 255

    return img

# generate model
def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5,5), (2,2), input_shape=(66,200,3), activation='elu'))
    model.add(Convolution2D(36, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(48, (5,5), (2,2), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))
    model.add(Convolution2D(64, (3,3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')

    return model

# training
def dataGenerator(imgPath, steeringList, batchSize, trainingFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imgPath) - 1)
            if trainingFlag:
                img, steering = augmentImage(imgPath[index], steeringList[index])
            else:
                img = mpimg.imread(imgPath[index])
                steering = steeringList[index]

            img = preprocessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))
