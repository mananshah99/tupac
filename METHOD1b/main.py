'''
Executes method one for overall classification. This file requires
    * a patch directory
    * a directory with associated mitotic heatmaps

Testing (July 5, 2016): Use 
    10 examples/1
    10 examples/2
    10 examples/3

Generate heatmaps for these images using googlenet, and generate feature vectors.
Make sure this is easily extensible to utilize all images
'''
import matplotlib
matplotlib.use('Agg')
import numpy as np
import mitosis_predict_regular as mp
from random import shuffle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

from quadratic_weighted_kappa import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mlxtend.classifier import EnsembleVoteClassifier

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

groundtruth = read_groundtruth()

### Define a dictionary of class : image number

mitosis_dictionary = {1 : [], 2 : [], 3 : []}
image_rna_dict = {i : [] for i in range(0, 501)}

for row in groundtruth:
    image_number  = row[0]
    image_mitosis = row[1]
    image_RNA     = row[2]
    
    # for now
    if int(image_number) < 500:
        mitosis_dictionary[int(image_mitosis)].append(image_number)
        image_rna_dict[int(image_number)].append(float(image_RNA))

### Iterate through the dictionary and select samples, or select all
SAMPLE_SIZE = -1 # 10
GEN_HEATMAPS = 1 # if this is set to 0, images that don't have corresponding heatmaps will be ignored (currently assigned [0...0] as feature vecs)

import random
image_ids = []
for key in mitosis_dictionary:
    tmp =  mitosis_dictionary[key] if SAMPLE_SIZE == -1 else random.sample(mitosis_dictionary[key], len(mitosis_dictionary[key]))[0:SAMPLE_SIZE]
    # this example is guaranteed to work
    
#    if key == 1:
#        tmp = ['001','006','008','009','010','014','015','016','017','018']
#    elif key == 2:
#        tmp = ['003','004','005','011','013','021','024','026','027','032']
#    elif key == 3:
#        tmp = ['007','012','019','023','029','030','036','041','046','047']
    
    image_ids.extend(tmp)

X = []
y = []

from tqdm import tqdm

patch_directory = 'patches_07-14-16'#'patches_06-29-16'

print "Obtaining classical features from patch directory " + patch_directory
bar = tqdm(total=len(image_ids))

for image_id in image_ids:
    import glob
    #where are the patches stored?
    globname = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/' + patch_directory + '/TUPAC-TR-' + image_id + '*'
    patches = []
    for patch_name in glob.glob(globname):
        patches.append(patch_name)

    image_level = 1 if (image_id in mitosis_dictionary[1]) else 2 if (image_id in mitosis_dictionary[2]) else 3
    
#    if image_level == 2:
#        bar.update(1)
#        continue

    features = mp.extract_features(patches)
#    print (image_id, features)

    if any(a == -1 for a in features):
        # print "Ignoring ", image_id
        bar.update(1)
        continue #ignore crap data

    image_level_RNA = image_rna_dict[int(image_id)][0]

    X.append(np.array(features, dtype=np.float32))
    y.append(image_level)
    bar.update(1)

bar.close()

X = np.vstack(X)
y = np.array(y, dtype=np.dtype(int))  #, dtype=np.float32)

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
clf2 = RandomForestClassifier(n_estimators=30, random_state=0)
clf2_1 = RandomForestClassifier(n_estimators=50, random_state=0)
clf2_2 = RandomForestClassifier(n_estimators=70, random_state=0)
clf3 = SVC(random_state=0, probability=True)

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf2_1, clf2_2, clf3], weights=[1,1.2,1.5,1.2,1], voting='soft')

from sklearn.metrics import *
ypred = []
print "Test ground truth ", ytst
for clf, lab, grd in zip([clf1, clf2, clf2_1, clf2_2, clf3, eclf],
                         ['Logistic Regression', 'Random Forest (30)', 'Random Forest (50)', 'Random Forest (70)', 'RBF kernel SVM', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
    clf.fit(Xtr, ytr)
    print lab
    ypred = clf.predict(Xtst)
    print ypred
    print classification_report(ytst, ypred)
    print confusion_matrix(ytst, ypred)
    print "Accuracy is ", accuracy_score(ytst, ypred) 
    print "Quadratic weighted kappa is ", kappa(ytst, ypred)

