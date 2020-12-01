import os, math
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from skimage.transform import resize, rotate
from skimage.util import random_noise

path = "normalized/test"
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', 'leq', 
               'neq', 'geq', 'alpha', 'beta', 'lambda', 'lt', 'gt', 'x', 'y']
nof_labels = len(label_names)
nof_images = 0

# get number of images
labels_dict = dict()
i = 0
for label in label_names:
    files = listdir(path + "/" + label)
    nof_images += len(files)
    labels_dict[label] = i
    i += 1
print("#nof_images: ", nof_images)
print(labels_dict)

images = np.zeros((nof_images, 48, 48), dtype=np.float32)
labels = np.zeros(nof_images, dtype=np.int)

i = 0
for label in label_names:
    files = listdir(path + "/" + label)
    label_no = labels_dict[label]
    
    for file in files:
        if i % 10000 == 0:
            print("At i=%d" % i)
        img = io.imread(path + "/" + label + "/" + file).astype(np.float32)
        img /= 255
               
        images[i] = img
        labels[i] = label_no

        i += 1
print("Finished")

plt.hist(labels, nof_labels)
plt.show()
plt.imshow(images[10000], cmap="gray")
plt.show()

np.save("test_images", images)
np.save("test_labels", labels)