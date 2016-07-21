import numpy as np
import sys
sys.path.insert(0, '../external/caffe/python')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import caffe

caffe.set_device(2)
caffe.set_mode_gpu()

# this uses the FULLY CONVOLUTIONAL roi detection model
model_def = 'deploy_fc.prototxt'
model_weights = 'models/cnn10_iter_227000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', np.array([104, 117, 123]))

f = open('../lists/mitosis.lst') #aim is to predict mitosis, not rna

i = 0
# THESE ARE ALL OF THE PATCHES

from tqdm import tqdm

bar = tqdm(total = sum(1 for line in open('../lists/mitosis.lst')))

for line in f:
    sample_image_name = line.strip('\n').split(' ')[0]
    sample_image_output = line.strip('\n').split(' ')[1]

    image = caffe.io.load_image(sample_image_name)
    image = caffe.io.resize_image(image, (1000, 1000))
    transformed_image = transformer.preprocess('data', image)

    net.blobs['data'].data[...] = transformed_image

    output = net.forward()

    sample_image_name = sample_image_name[sample_image_name.index('TUPAC-TR-'):-4]
    a = output['prob']

    i += 1

    # (1, 2, 57, 57), the positive one
    im = a[0][0]
    
    plt.imsave('../pos_result_maps/' + sample_image_name + '_class=' + str(sample_image_output) + '.png', im)
    bar.update(1)
bar.close()
