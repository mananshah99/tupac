import matplotlib
matplotlib.use('Agg')

import os
from skimage.io import *
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

TYPE = "tumor"

PATH = TYPE + "_06-21-16"
OUT =  TYPE + "_06-21-16-color"

files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

for tmp in files:
    if os.path.isfile(OUT + "/" + str(tmp)):
        print("Already seen " + str(tmp))
        continue
    else:
        print("Converting " + str(tmp))

    image = imread(PATH + "/" + tmp)
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(image)
    rgba_img = np.delete(rgba_img, 3, 2)
    plt.imshow(rgba_img)
    plt.savefig(OUT + "/" + str(tmp))
