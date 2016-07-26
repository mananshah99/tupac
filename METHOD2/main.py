'''
Executes method two for overall classification
'''

import sys
sys.path.insert(0, '../METHOD3/external/caffe/python')
import caffe

import matplotlib
matplotlib.use('Agg')
import numpy as np
import extract_cnn_features
from random import shuffle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

from quadratic_weighted_kappa import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier

from sklearn.cluster import KMeans

def read_groundtruth(filename = 'training_ground_truth.csv'):
    import csv
    output = [] # format: IMAGE_NAME (001), CLASS (1), RNA (-0.3534)
    
    with open(filename, 'rb') as f:
        rownum = 1
        reader = csv.reader(f)
        for row in reader:
            row.insert(0, str(rownum).zfill(3))
            rownum += 1
            output.append(row)
    
    return output

def generate_input_list(samples_per_class = 10):
    print '__implement__'

groundtruth = read_groundtruth()

### Define a dictionary of class : image number

mitosis_dictionary = {1 : [], 2 : [], 3 : []}

for row in groundtruth:
    image_number  = row[0]
    image_mitosis = row[1]
    image_RNA     = row[2]
    
    # for now
    #if image_number < '100': 
    #    mitosis_dictionary[int(image_mitosis)].append(image_number)

### Iterate through the dictionary and select samples, or select all
SAMPLE_SIZE = -1 # 10
GEN_HEATMAPS = 1 # if this is set to 0, images that don't have corresponding heatmaps will be ignored

import random
image_ids = []
for key in mitosis_dictionary:
    tmp =  mitosis_dictionary[key] if SAMPLE_SIZE == -1 else random.sample(mitosis_dictionary[key], len(mitosis_dictionary[key]))[0:SAMPLE_SIZE]
    
    # this example is guaranteed to work
    
    if key == 1:
        tmp = ['001','006','008','009','010','014','015','016','017','018']
    elif key == 2:
        tmp = ['003','004','005','011','013','021','024','026','027','032']
    elif key == 3:
        tmp = ['007','012','019','023','029','030','036','041','046','047']
    

    image_ids.extend(tmp)

X = []
y = []

caffe.set_device(2)
caffe.set_mode_gpu()

model_def = 'deploy.prototxt'
model_weights = 'mitosis_stage1.caffemodel' # this is the caffemodel of the MITOSIS predictor (used to generate the mitosis heatmaps)

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', np.array([150, 150, 150])) #np.array([104, 117, 123]))
net.blobs['data'].reshape(1, 3, 63, 63) #224, 224) # -- mitko's model is 63x63

patch_directory = 'patches_07-14-16'#'patches_06-29-16'

print "Executing phase 1 -- Caffe feature extraction & KMeans"
##### FIRST PHASE: FIT KMEANS

image_ids = image_ids[0:5] + image_ids[10:15] + image_ids[-5:]

from tqdm import tqdm
bar = tqdm(total=len(image_ids))

for image_id in image_ids:
    import glob
    globname = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/' + patch_directory + '/TUPAC-TR-' + image_id + '*'
    patches = []
    for patch_name in glob.glob(globname):
        patches.append(patch_name)

    features = extract_cnn_features.extract_features(net, transformer,  patches)
    image_level = 1 if (image_id in mitosis_dictionary[1]) else 2 if (image_id in mitosis_dictionary[2]) else 3

    X.extend(features) 
    y.append(image_level)
#    print (image_id, len(features))
    bar.update(1)

bar.close()
 
kmeans_input = []
for i in range(0, len(X), 1024):
    kmeans_input.append(X[i : i +  1024])

N_CLUSTERS = 50

km = KMeans(n_clusters=N_CLUSTERS, n_jobs=-1)
km.fit(kmeans_input)

print "Executing phase 2 -- Bin counting (bag-of-words)"
##### SECOND PHASE: COUNT BINS

X_norm = []
y_norm = []
bar = tqdm(total=len(image_ids))

for image_id in image_ids:
    import glob
    globname = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/' + patch_directory + '/TUPAC-TR-' + image_id + '*'
    patches = []
    for patch_name in glob.glob(globname):
        patches.append(patch_name)

    features = extract_cnn_features.extract_features(net, transformer,  patches)
    image_level = 1 if (image_id in mitosis_dictionary[1]) else 2 if (image_id in mitosis_dictionary[2]) else 3

    histogram = [0 for i in xrange(N_CLUSTERS)]
    kmeans_input = []
    for i in range(0, len(features), 1024):
        kmeans_input.append(features[i : i +  1024])

    try:
        pred = km.predict(kmeans_input)

        for p in pred:
            histogram[p] += 1
   
    except:
        pass

    features_normalized = histogram 
    
    X_norm.append(features_normalized)
    y_norm.append(image_level)

#    print (image_id, features_normalized)
    bar.update(1)

bar.close()

X = np.vstack(X_norm)
y = np.array(y_norm)

#### Perform prediction. X and y are the arrays which are split into tr and tst

import random

indices = [i for i in range(0, len(X))]
shuffle(indices)

Xtr = X[indices[0:int(len(indices) * 0.8)]]
ytr = y[indices[0:int(len(indices) * 0.8)]]

Xtst = X[indices[int(len(indices) * 0.8):]]
ytst = y[indices[int(len(indices) * 0.8):]]

print ("Xtr, ytr, Xtst, ytst")
print (Xtr.shape, ytr.shape, Xtst.shape, ytst.shape)

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(n_estimators=50, random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 1, 1], voting='soft')

from sklearn.metrics import *

ypred = []
for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
    clf.fit(Xtr, ytr)
    print lab
    ypred = clf.predict(Xtst)
    print ytst
    print ypred
    print classification_report(ytst, ypred)
    print kappa(ytst, ypred)
    # kappa(rater_a, rater_b, min_rating=None, max_rating=None)
