import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.io import *
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import cv2

TYPE = "tumor"
PATH = TYPE + "_06-21-16"
files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

ROI_IMAGES =  ['356', '147', '296', '244', '310', '056', '121', '197', '184', '084', '102', '288', '354', '431', '416', '137', '353', '231', '377', '225', '086', '228', '033', '190', '308', '393', '403', '327', '054', '439', '123', '078', '369', '049', '132', '275', '227', '466', '139', '144', '094', '303', '089', '257', '440', '152', '460', '390', '233', '252', '293', '189', '425', '284', '266', '375', '220', '341', '176', '246', '159', '172', '462', '358', '241', '332', '335', '382', '216', '406', '113', '451', '456', '150', '337', '230', '186', '074', '060', '267', '338', '350', '207', '345', '391', '326', '235', '125', '316', '276', '211', '483', '203', '047', '073', '272', '212', '050', '421', '426', '427', '455', '253', '351', '196', '468', '396', '435', '154', '065', '330', '072', '115', '148', '210', '445', '200', '360', '298', '261', '187', '444', '181', '459', '370', '178', '051', '475', '052', '191', '204', '365', '111', '239', '392', '040', '214', '238', '262', '453', '339', '221', '138', '032', '258', '128', '449']

print len(ROI_IMAGES)

ROI_IMAGES_FULL_NAMES = ['TUPAC-TR-' + i + '.png' for i in ROI_IMAGES]

DATA_LOCATION = '/data/dywang/Database/Proliferation/data/'

def test_tumor_roi():
    for f in files:
        if f in ROI_IMAGES_FULL_NAMES:
            plotIndsCV(f, DATA_LOCATION, 0)


def plotIndsCV(fname, dataLoc, inputLevel):
    
    image = cv2.imread(join(PATH, fname))
    ROIs = np.genfromtxt(dataLoc + 'ROI/ROI/' + fname[:-4] + '-ROI.csv', delimiter=",")
    numRegions = len(ROIs[:,0])
    print join(PATH, fname)

    for i in range(0, numRegions):
        x, y, w, h = to_higher_level(0, 2, ROIs[i,:])
        topLeft = (x, y)
        bottomRight = (x + w, y + h)
        cv2.rectangle(image, topLeft, bottomRight, (255, 255, 255), 3)
    
    cv2.imwrite('test_roi/' + fname, image) 

def plotInds(fname, dataLoc, inputLevel):
    
    test_im = Image.open(join(PATH, fname))
    width, height = test_im.size    

    plt.axis('off')

    print width
    print height
    
    MY_DPI = 129

    fig = plt.figure(figsize=(width/MY_DPI, height/MY_DPI), dpi=MY_DPI, frameon=False)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ROIs = np.genfromtxt(dataLoc + 'ROI/ROI/' + fname[:-4] + '-ROI.csv', delimiter=",")
    numRegions = len(ROIs[:,0])
    print join(PATH, fname) 

    ax.imshow(imread(join(PATH, fname)))

    for i in range(0, numRegions):
        x, y, w, h = to_higher_level(0, 2, ROIs[i,:])
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor="black"
            )
        )
   
    fig.savefig('test_roi/' + fname, dpi=MY_DPI)

def to_higher_level(in_level, out_level, values):
    newvalues = []
    scale = np.power(4, out_level - in_level)
    for value in values:
        newvalues.append(int(value/scale))
    return newvalues

def genInds(dataLoc, ID, inputLevel):
    ROIs = np.genfromtxt(dataLoc + 'ROI/ROI/TUPAC-TR-' + ID + '-ROI.csv', delimiter=",")

    numRegions = len(ROIs[:,0])
    ind_h = []
    ind_w = []

    for i in range(0, numRegions):
        cur_x, cur_y, cur_w, cur_h = to_higher_level(0, 2, ROIs[i,:])

        for j in range(cur_x, cur_x + cur_w):
            for k in range(cur_y, cur_y + cur_h):
                ind_h.append(k)
                ind_w.append(j)

    return np.asarray(ind_h), np.asarray(ind_w)

test_tumor_roi()
