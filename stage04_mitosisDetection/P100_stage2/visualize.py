# Visualization file for mitosis detection 
#   1) AUROC + ROC curve
#   2) Filter visualization
#
# To modify, change the model_def and model_weights
#

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
sys.path.insert(0, '/home/dywang/dl/caffe/python')

import caffe

caffe.set_mode_gpu()

model_def = 'deploy.prototxt'
model_weights = 'models/gnet_iter_140000.caffemodel' #'models/LEVEL1/cnn10_iter_180000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', np.array([104, 117, 123]))
net.blobs['data'].reshape(100, 3, 224, 224)

f = open('../training_examples/val.lst')

from sklearn import metrics

def get_auc(batch_sz = 100):
    y_true = []
    y_score = []
    i = 0

    lines = f.readlines()
    im_lines = [line.strip('\n').split(' ')[0] for line in lines]
    y_actual = [int(line.strip('\n').split(' ')[1]) for line in lines]

    im_lines = im_lines[0:5000] + im_lines[len(im_lines)-5000:]
    y_actual = y_actual[0:5000] + y_actual[len(y_actual)-5000:]

    for i in range(0, int(len(im_lines)/float(batch_sz))):
        im_lines_i = im_lines[i * batch_sz : (i + 1) * batch_sz]
        transformed_images = [transformer.preprocess('data', caffe.io.resize_image(caffe.io.load_image(im_name), (224,224))) for im_name in im_lines_i]
        batch = np.array(transformed_images)

        net.blobs['data'].data[...] = batch 

        ### perform classification
        output = net.forward()
    
        output_prob = output['prob']  # the output probability vector for the first image in the batch
        # prob of positive
        for i in range(0, len(transformed_images)):
            pos_prob = output_prob[i][1] #[i][1][0][0]
            y_score.append(pos_prob)

    print metrics.roc_auc_score(y_actual, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_actual, y_score)

    plt.scatter(fpr, tpr)
    plt.savefig('ROC.png')

# get_auc()
        
f = open('../training_examples/val.lst')
sample_image_name = f.readline().strip('\n').split(' ')[0]
sample_image_output = open('../training_examples/val.lst').readline().strip('\n').split(' ')[1]

image = caffe.io.load_image(sample_image_name)
image = caffe.io.resize_image(image, (224,224))

print image.shape
transformed_image = transformer.preprocess('data', image)

net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

#print 'predicted class is:', output_prob.argmax()
#print 'actual class is:', sample_image_output

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    print data.shape
    plt.axis('off')
 
    plt.imsave('map.png', data)

# helper show filter outputs
def show_filters(net):
    net.forward()
    plt.figure()
    NAME = 'inception_4e/3x3'
    filt_min, filt_max = net.blobs[NAME].data.min(), net.blobs[NAME].data.max()
    for i in range(3):
        plt.subplot(1,4,i+2)
        plt.title("filter #{} output".format(i))
        plt.tight_layout()
        plt.axis('off')
        plt.imsave('filter #{}'.format(i) + ' .png', net.blobs[NAME].data[0, i], vmin=filt_min, vmax=filt_max)

# filter the image with initial 

show_filters(net)

# the parameters are a list of [weights, biases]
#filters = net.params['conv11'][0].data
#print filters.shape
#print filters[0][0].shape
#plt.imsave('map2.png', filters[0][0])
#vis_square(filters.transpose(0, 2, 3, 1)) 
