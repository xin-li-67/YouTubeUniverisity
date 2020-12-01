import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Params
path = "myData"
labelFile = 'labels.csv'
batch_size_val = 50
steps_per_epoch_val = 400
epoches_val = 10
imageDimensions = (32, 32, 3)
test_ratio = 0.2
validation_ratio = 0.2

# Import
count = 0
images = []
classNo = []
myList = os.listdir(path)

print("Total Classes Detected:", len(myList))

noOfClasses = len(myList)

print("Importing Classes.....")

for x in range (0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))

    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
        
    print(count, end = " ")
    count += 1

print(" ")

images = np.array(images)
classNo = np.array(classNo)

# Split data
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_ratio)
# x_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID

# Check if the quantity of images matches the quantity of the labels for each dataset
print("Data Shapes")
print("Train", end = "")
print(x_train.shape, y_train.shape)
print("Validation", end = "")
print(x_validation.shape, y_validation.shape)
print("Test", end = "")
print(x_test.shape, y_test.shape)

assert(x_train.shape[0] == y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(x_validation.shape[0] == y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert(x_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(x_train.shape[1:] == (imageDimensions))," The dimesions of the Training images are wrong "
assert(x_validation.shape[1:] == (imageDimensions))," The dimesionas of the Validation images are wrong "
assert(x_test.shape[1:] == (imageDimensions))," The dimesionas of the Test images are wrong"

# Read csv
data = pd.read_csv(labelFile)
print('Data shape', data.shape, type(data))

# Display some sample images
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()

for i in range(cols):
    for j, row in data.iterrows():
        x_selected = x_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

# Bar chart
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# Preprocessing
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def equalize(img):
    img = cv2.equalizeHist(img)

    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255 # normalize values

    return img

x_train = np.array(list(map(preprocessing, x_train)))
x_validation = np.array(list(map(preprocessing, x_validation)))
x_test = np.array(list(map(preprocessing, x_test)))
cv2.imshow("Grayscale Images", x_train[random.randint(0, len(x_train) - 1)])

# Add a depth of 1
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

# Image augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(x_train)
batches = dataGen.flow(x_train, y_train, batch_size=20)
x_batch, y_batch = next(batches)

fig,axs = plt.subplots(1, 15, figsize=(20,5))
fig.tight_layout()
 
for i in range(15):
    axs[i].imshow(x_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
    axs[i].axis('off')

plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Convolutional neural network model
def trainModele():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
 
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = trainModele()
print(model.summary())

# Training process
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size_val), steps_per_epoch=steps_per_epoch_val, 
                              epochs=epoches_val, validation_data=(x_validation, y_validation), shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)

print("Test Score = ", score[0])
print("Test Accuracy = ", score[1])

# Save the model
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()