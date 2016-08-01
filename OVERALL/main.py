import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import extract_patch as extract_patch 
import extract_wsi as extract_wsi
import sys

import random
from random import shuffle
import numpy as np

from utils import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

from sklearn import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from quadratic_weighted_kappa import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mlxtend.classifier import EnsembleVoteClassifier
from tqdm import tqdm
import glob

parser = argparse.ArgumentParser(description="Perform prediction for TUPAC 2016 results given image features")

parser.add_argument('feature_directory', help="The directory where the image files are stored (corresponding to the TUPAC training data). Generally, image files may be softmax outputs, whole slide image heatmaps, or patch heatmaps.")
parser.add_argument('--patch', dest='ispatch', action='store_true', help="Pass this argument if the input feature data are image patches (not WSI heatmaps). Default is patch")
parser.add_argument('--no-patch', dest='ispatch', action='store_false', help="Pass this argument if the input feature data are WSI heatmaps (not image patches). Default is patch")
parser.set_defaults(feature=True)

parser.add_argument('-d', '--directory', help="The directory where patches are stored (if patch is True)", default="")  
parser.add_argument('-m', '--mode', help="Whether to make predictions for mitosis (type='mitosis') or rna (type='rna'). Default is mitosis",  default="mitosis")
parser.add_argument('-r', '--seed', help="Random seed for result reproducibility.  Default is 0. Enter -1 for no seed.", default=0)
parser.add_argument('-s', '--split', help="Train/test split percentage. Default is 0.2 (80/20 split)", default=0.2)
parser.add_argument('-p', '--portion', help="Sample portion from each class, or select all. Default is selecting all (-1)", default=-1)

# Set up dictionaries and select image IDs for sampling
args = parser.parse_args()

if args.seed != -1:
    np.random.seed(args.seed)
    random.seed(args.seed)

processed = process_groundtruth()

class_dictionary  = {1 : [], 2 : [], 3 : []}
ground_dictionary = {i : [] for i in range(1, 501)}

for row in processed:
    image_number  = int(row[0])
    image_mitosis = int(row[1])
    image_RNA     = float(row[2])
    
    class_dictionary[image_mitosis].append(row[0])
    ground_dictionary[image_number].extend([image_mitosis, image_RNA])

rna_list = []
for key in ground_dictionary:
    rna = ground_dictionary[key][1]
    rna_list.append(rna)

sort(rna_list)

SAMPLE_SIZE = int(args.portion)

image_ids = []
for key in class_dictionary:
    tmp =  class_dictionary[key] if SAMPLE_SIZE == -1 else random.sample(class_dictionary[key], len(class_dictionary[key]))[0:SAMPLE_SIZE]
    
    #if key == 1:
    #    tmp = ['001','006','008','009','010','014','015','016','017','018']
    #elif key == 2:
    #    tmp = ['003','004','005','011','013','021','024','026','027','032']
    #elif key == 3:
    #    tmp = ['007','012','019','023','029','030','036','041','046','047']
    
    image_ids.extend(tmp)

print "Using ", len(image_ids), " ids, with SAMPLE_SIZE set to ", args.portion
X = []
y = []

# If we are looking at patches, process the patches
if args.ispatch:
    patch_directory = args.directory
    print "Patch argument selected. Obtaining images from patch directory ", patch_directory

    bar = tqdm(total=len(image_ids))
    for image_id in image_ids:

        #where are the patches stored?
        globname =  patch_directory + '/TUPAC-TR-' + image_id + '*'

        patches = []
        for patch_name in glob.glob(globname):
            patches.append(patch_name)

        if args.mode == 'mitosis':
            image_level = ground_dictionary[int(image_id)][0] 
        else:
            image_level = ground_dictionary[int(image_id)][1]

        features = extract_patch.extract_features(args.feature_directory, patches)
        #print (image_id, features)

        if any(a == -1 for a in features):
            bar.update(1)
            continue #ignore crap data

        X.append(preprocessing.scale(np.array(features, dtype=np.float32)))
        y.append(image_level)
        bar.update(1)

    bar.close()
else:
    heatmap_directory = args.feature_directory
    print "WSI argument selected. Obtaining images from directory ", heatmap_directory
    
    bar = tqdm(total=len(image_ids))
    for image_id in image_ids:
        name = heatmap_directory + '/TUPAC-TR-' + image_id + '.png'

        if args.mode == 'mitosis':
            image_level = ground_dictionary[int(image_id)][0]
        else:
            image_level = ground_dictionary[int(image_id)][1]

        features = extract_wsi.extract_features(name)
        #print (image_id, features)
        #print (image_id, len(features))

        if any(a == -1 for a in features):
            bar.update(1)
            continue #ignore crap data

        X.append(preprocessing.scale(np.array(features, dtype=np.float32)))
        y.append(image_level)
        bar.update(1)
    bar.close()

X = np.vstack(X)

if args.mode == 'mitosis':
    y = np.array(y, dtype=np.dtype(int))  #, dtype=np.float32)
else:
    y = np.array(y, dtype=np.float32)

if args.seed != -1:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=args.split, random_state=args.seed)
else:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=args.split)

if args.mode == 'mitosis': # classification problem

    print "[PREDICTION > SVM] Tuning via Grid Search"

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 100, 1000, 5000, 10000]}]

    scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("\t Best parameters set found on development set:")
        print(clf.best_params_)
        print("\t Grid scores on development set:")
        
        for params, mean_score, scores in clf.grid_scores_:
            print("\t %0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

        print("\t Detailed classification report:")
        print("\t The model is trained on the full development set.")
        print("\t The scores are computed on the full evaluation set.")
    
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

    print "[PREDICTION > RF] Tuning via Grid Search"
    
    tuned_parameters = {"max_depth": [10, None],
              "min_samples_split": [2, 10],
              "min_samples_leaf": [1, 10],
              "n_estimators": [60, 70, 80],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(RandomForestClassifier(n_estimators=50), tuned_parameters, cv=5,
                       scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("\t Best parameters set found on development set:")
        print(clf.best_params_)
        print("\t Grid scores on development set:")
        
        for params, mean_score, scores in clf.grid_scores_:
            print("\t %0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

        print("\t Detailed classification report:")
        print("\t The model is trained on the full development set.")
        print("\t The scores are computed on the full evaluation set.")
    
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
else:
    # rna!
    print "[REGRESSION > RF] Tuning via Grid Search"

    tuned_parameters = {"max_depth": [10, None],
              "min_samples_split": [2, 10],
              "min_samples_leaf": [1, 10],
              "n_estimators": [60, 70, 80],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(RandomForestRegressor(n_estimators=50), tuned_parameters, cv=5,
                       scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("\t Best parameters set found on development set:")
        print(clf.best_params_)
        print("\t Grid scores on development set:")

        for params, mean_score, scores in clf.grid_scores_:
            print("\t %0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

        print("\t Detailed classification report:")
        print("\t The model is trained on the full development set.")
        print("\t The scores are computed on the full evaluation set.")

        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

