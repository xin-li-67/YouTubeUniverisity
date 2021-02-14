import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

from Latex.Latex import Latex
from skimage import io
from IPython.display import display, Math

mean_train = np.load("train_images_mean.npy")
std_train = np.load("train_images_std.npy")
model = Latex("model", mean_train, std_train, plotting=False)
ltokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '=', '#leq', '#neq', 
           '#geq', '#alpha', '#beta', '#lambda', '#lt', '#gt', 'x', 'y', '^', '#frac', '{', '}' ,' ']

files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("formulas/train")) for f in fn]
nlabels = 23
nof_sequences = len(files)
nclasses = nlabels + 4 + 2 + 1 # 1 for pad and 4 for relative pos, 2 for abs pos and shift from last and width
noclasses = len(ltokens)

isequence = np.zeros((nof_sequences, 30, nclasses))
isequence[:, :, -1] = 1 # default all pad
osequence = (noclasses - 1) * np.ones((nof_sequences, 30), dtype=int)
osequence[:, 0] = noclasses # all <GO> in beginning

for i in range(nof_sequences):
    if i % 10 == 0:
        print("Start i: %d" % i)
    
    formula = io.imread(files[i])
    pos = files[i].rfind("/")
    filename = files[i][pos + 1:]
    height, width = formula.shape
    correct = model.filename2formula(filename)
    oseq = model.filename2seq(filename)
    osequence[i, 1:len(oseq) + 1] = oseq
    latex = model.predict(formula)
    
    last_xmax = 0
    last_ymin = latex['data'][0]['ymin']
    step_c = -1

    for step in latex['data']:
        step_c += 1
        isequence[i][step_c][:nlabels] = step['probs']
        isequence[i][step_c][-1] = 0 # remove pad
        isequence[i][step_c][-7] = step['xmin'] / width
        isequence[i][step_c][-6] = step['ymin'] / height
        isequence[i][step_c][-5] = (step['xmin'] - last_xmax) / 10
        last_xmax = step['xmax']
        isequence[i][step_c][-4] = (step['xmax'] - step['xmin']) / 48
        isequence[i][step_c][-3] = (step['ymin'] - last_ymin) / 10
        isequence[i][step_c][-2] = (step['ymax'] - step['ymin']) / 48
        last_ymin = step['ymin']

np.save("iseq_n", isequence)
np.save("oseq_n", osequence)
with open("files.json", "w") as f:
    f.write(json.dumps(files[:nof_sequences]))