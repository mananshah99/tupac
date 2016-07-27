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

caffe.set_device(1)
caffe.set_mode_gpu()

# this uses the FULLY CONVOLUTIONAL roi detection model
model_def = 'deploy_full_net.prototxt'
model_weights = 'models-resultlayer_iter_64500.caffemodel' #/cnn10_iter_227000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', np.array([150, 150, 150]))

f = open('../lists/mitosis_val.lst') #aim is to predict mitosis, not rna

i = 0
# THESE ARE ALL OF THE PATCHES

from tqdm import tqdm

bar = tqdm(total = 200) #sum(1 for line in open('../lists/mitosis.lst')))

y_pred = []
y_true = []
idx = 0
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

    #print a[0,:,0,0].shape
    probs = a[0,:,0,0]
    prediction = probs.argmax(axis=0)
    true_value = int(sample_image_output)
    
    if true_value == 1:
        continue
    else:
        if prediction == 1:
            prediction = 0
        if prediction > 1:
            prediction -= 1
        if true_value > 1:
            true_value -= 1
        y_pred.append(prediction)
        y_true.append(true_value)
        idx += 1
        bar.update(1)
    if idx >= 200:
        break

bar.close()

print y_true
print y_pred
from sklearn.metrics import *
print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Precision: ", precision_score(y_true, y_pred))
print("Classification Report")
print classification_report(y_true, y_pred)
print("Confusion Matrix")
print confusion_matrix(y_true, y_pred)
