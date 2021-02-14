import os
from sklearn.model_selection import train_test_split
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
path = 'materials'
data = importDataInfo(path)

data = dataBalancer(data, display=True)
imgPath, steerings = loadData(path, data)

xTrain, xVal, yTrain, yVal = train_test_split(imgPath, steerings, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

model = createModel()

history = model.fit(dataGenerator(xTrain, yTrain, 100, 1), steps_per_epoch=100, epochs=10, validation_data=dataGenerator(xVal, yVal, 50, 0), validation_steps=50)

model.save('model.h5')
print('Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()