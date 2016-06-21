from __future__ import print_function, division

import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage as ski
import skimage.morphology as morp

import pickle
import functools
import operator
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from sklearn.metrics import confusion_matrix

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

# msktype is 0 if it's a simple addition
# msktype is 1 if it's an overlay
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

def gaussian_filter(kernel_shape):
    x = np.zeros((kernel_shape, kernel_shape), dtype=theano.config.floatX)

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma**2
        return  1./Z * np.exp(-(x**2 + y**2) / (2. * sigma**2))

    for i in xrange(kernel_shape):
        for j in xrange(kernel_shape):
            x[i,j] = gauss(i-4., j-4.)

    return x / np.sum(x)


def lecun_lcn(input, img_shape, kernel_shape, threshold=1e-4):
    """
    Yann LeCun's local contrast normalization
    This is performed per-colorchannel!!!

    http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
    """
    input = input.reshape((input.shape[0], 1, input.shape[1], input.shape[2]))
    X = T.matrix(dtype=input.dtype)
    X = X.reshape((len(input), 1, img_shape[0], img_shape[1]))

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = theano.shared(gaussian_filter(kernel_shape).reshape(filter_shape))

    convout = conv.conv2d(input=X,
                          filters=filters,
                          image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                          filter_shape=filter_shape,
                          border_mode='full')

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape / 2.))
    centered_X = X - convout[:, :, mid:-mid, mid:-mid]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv.conv2d(input=T.sqr(X),
                             filters=filters,
                             image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                             filter_shape=filter_shape,
                             border_mode='full')

    denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
    per_img_mean = T.mean(denom, axis=(1, 2))
    divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
    divisor = T.maximum(divisor, threshold)

    new_X = centered_X / divisor
    #new_X = theano.tensor.flatten(new_X, outdim=3)

    f = theano.function([X], new_X)
    return f(input)


def lcn_image(images, kernel_size=9):
    """
    This assumes image is 01c and the output will be c01 (compatible with conv2d)

    :param image:
    :param inplace:
    :return:
    """
    image_shape = (images.shape[1], images.shape[2])
    if len(images.shape) == 3:
        # this is greyscale images
        output = lecun_lcn(images, image_shape, kernel_size)
    else:
        # color image, assume RGB
        r = images[:, :, :, 0]
        g = images[:, :, :, 1]
        b = images[:, :, :, 2]

        output = np.concatenate((
            lecun_lcn(r, image_shape, kernel_size),
            lecun_lcn(g, image_shape, kernel_size),
            lecun_lcn(b, image_shape, kernel_size)),
            axis=1
        )
    return output


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """ Code adopted from here: https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/preprocessing.py
        but can work with b01c and bc01 orderings

        An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim > 2, "X.ndim must be more than 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    aggr_axis = tuple(np.arange(len(X.shape) - 1) + 1)
    mean = np.mean(X, axis=aggr_axis, keepdims=True)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + np.var(X, axis=aggr_axis, ddof=ddof, keepdims=True)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + np.sum((X ** 2), axis=aggr_axis, keepdims=True)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X
