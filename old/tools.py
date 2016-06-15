from __future__ import print_function, division

import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage as ski
import skimage.morphology as morp

def img_show(img, title=None, margin=0.05, dpi=400, cmap = 'gray'):
    nda = img
    spacing = [1 ,1, 1]

    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap(cmap)
    ax.imshow(nda, extent=extent, interpolation=None)
    if title:
        plt.title('%s' % (title))
    return ax

def img_show_with_rect(img, rect =[10,10,20,20], ax = None, title=None, margin=0.05, dpi=400):
    import matplotlib.patches as patches
    if ax == None:
        ax = img_show(img, title=None, margin=0.05, dpi=dpi)

    ax.add_patch(
        patches.Rectangle(
            (rect[0], rect[1]),   # (x,y)
            rect[2],          # width
            rect[3],          # height
            edgecolor = 'r',
            fill=False
        )
    )
    return ax

def findROI(img):
    img2 = ski.color.rgb2hsv(img)
    msk0v = ski.filters.threshold_otsu(img2[:,:,0])
    msk1v = ski.filters.threshold_otsu(img2[:,:,1])

    msk0 = img2[:,:,0] > msk0v
    msk1 = img2[:,:,1] > msk1v

    msk0 = morp.remove_small_objects(msk0, min_size=36)
    msk1 = morp.remove_small_objects(msk1, min_size=36)

    msk3 = np.logical_and(msk0, msk1)
    disk = morp.disk(20)
    msk3 = morp.binary_dilation(msk3, disk)

    msk = ski.img_as_int(msk3)
    return msk

def addMask(img, msk, color):
    cts = ski.measure.find_contours(msk, 0)
    for ct in cts:
        ct = ct[:, [1, 0]]
        pts = ct.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, 5)
    return img

def addMaskOverlay(img, msk, color):
#    msk = msk.astype(np.uint8)
    alpha = 0.7
#    msk_jet = cv2.applyColorMap(msk, cv2.COLORMAP_JET)
    img_overlay = cv2.addWeighted(img, alpha, msk, 1 - alpha, 1);
    return img_overlay

def doAddMask(imgName, mskName, color, outputName, msktype = 0):
    img = cv2.imread(imgName)
    if msktype == 0:
        msk = ski.io.imread(mskName, True) > 0
        img = addMask(img, msk, color)
    elif msktype == 1:
        msk = cv2.imread(mskName)
        img = addMaskOverlay(img, msk, color)
    cv2.imwrite(outputName, img)

def doAddMasks(imgName, mskName_list, color_list, outputName):
    img = cv2.imread(imgName)
    for mskName, color in zip(mskName_list, color_list):
        msk = ski.io.imread(mskName, True) > 0
        img = addMask(img, msk, color)
    cv2.imwrite(outputName, img)
