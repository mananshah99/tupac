import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import extract_patch as extract_patch 
import extract_wsi as extract_wsi
import sys
import os
import datetime

import random
from random import shuffle
import numpy as np

from utils import *
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from scipy.stats import spearmanr
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
import cPickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description="Perform prediction for TUPAC 2016 results given image features")

parser.add_argument('feature_directory', help="The directory where the image files are stored (corresponding to the TUPAC training data). Generally, image files may be softmax outputs, whole slide image heatmaps, or patch heatmaps.")
parser.add_argument('--patch', dest='ispatch', action='store_true', help="Pass this argument if the input feature data are image patches (not WSI heatmaps). Default is patch")
parser.add_argument('--no-patch', dest='ispatch', action='store_false', help="Pass this argument if the input feature data are WSI heatmaps (not image patches). Default is patch")
parser.set_defaults(feature=True)

parser.add_argument('-d', '--directory', help="The directory where patches are stored (if patch is True)", default="")  
parser.add_argument('-m', '--mode', help="Whether to make predictions for mitosis (type='mitosis') or rna (type='rna'). Default is mitosis",  default="mitosis")
parser.add_argument('-r', '--seed',type=int, help="Random seed for result reproducibility.  Default is 0. Enter -1 for no seed.", default=0)
parser.add_argument('-s', '--split', type=int, help="Train/test split percentage. Default is 0.2 (80/20 split)", default=0.2)
parser.add_argument('-p', '--portion', type=int, help="Sample portion from each class, or select all. Default is selecting all (-1)", default=-1)
parser.add_argument('-e', '--experiments', type=int, help="Generate experiment plots (oob error rate varying trees, feature importances). Default is 0", default=0)
parser.add_argument('-k', '--pickle', help="Pickle outputs for re-use (generally do this when running on entire dataset, WILL OVERWRITE last save). Default is 0", default=0)
parser.add_argument('-l', '--load', type=int, help="Load from previous pickle file (X and y, must change from code default)", default=0)
# Set up dictionaries and select image IDs for sampling
args = parser.parse_args()

mydir = os.path.join(os.getcwd(), "results/" + args.mode.upper() + "-" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

try:
    os.makedirs(mydir)
except OSError, e:
    if e.errno != 17:
        raise # This was not a "directory exist" error..

print "Saving all output to directory ", mydir
logfile = open(os.path.join(mydir, 'output.txt'), 'wb')  # File where you need to keep the logs

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = logfile

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

print "Using arguments"
print(args)

if args.seed != -1:
    seed = int(args.seed)
    np.random.seed(seed)
    random.seed(seed)

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

rna_list.sort()

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

#print "Class balances"
#for key in class_dictionary:
#    print (key, len(class_dictionary[key]))

print "Using ", len(image_ids), " ids, with SAMPLE_SIZE set to ", args.portion

X = []
y = []

def process_images_patch(image_id):
    global args
    global ground_dictionary
    
    patch_directory = args.directory
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

    return np.array(features), image_level #preprocessing.scale(np.array(features, dtype=np.float32)), image_level

def process_images_nopatch(image_id):
    print "\t Beginning ", image_id
    global args
    global ground_dictionary
    
    heatmap_dictionary = args.feature_directory
    name = heatmap_directory + '/TUPAC-TR-' + image_id + '.png'

    if args.mode == 'mitosis':
        image_level = ground_dictionary[int(image_id)][0]
    else:
        image_level = ground_dictionary[int(image_id)][1]

    features = extract_wsi.extract_features(name)
    #print (image_id, features)
    #print (image_id, len(features))

    print "\t Completed ", image_id
    return np.array(features, dtype=np.float32), image_level

#    return preprocessing.scale(np.array(features, dtype=np.float32)), image_level

if args.load == 0:
    # If we are looking at patches, process the patches
    if args.ispatch:
        patch_directory = args.directory
        print "Patch argument selected. Obtaining images from patch directory ", patch_directory
        pool = mp.Pool(processes = 30)
        X_and_y = pool.map(process_images_patch, image_ids)
        X, y = zip(*X_and_y)
    
    else:
        heatmap_directory = args.feature_directory
        print "WSI argument selected. Obtaining images from directory ", heatmap_directory
        pool = mp.Pool(processes = 30)
        X_and_y = pool.map(process_images_nopatch, image_ids)
        X, y = zip(*X_and_y)

    X = np.vstack(X)

    if args.mode == 'mitosis':
        y = np.array(y, dtype=np.dtype(int))  #, dtype=np.float32)
    else:
        y = np.array(y, dtype=np.float32)

    if int(args.pickle) == 1:
        with open(os.path.join(mydir, 'X.pickle'), 'wb') as f:
            cPickle.dump(X, f)

        with open(os.path.join(mydir, 'y.pickle'), 'wb') as f:
            cPickle.dump(y, f)

else:
    # MITKONET
    X = cPickle.load(open('results/MITOSIS-2016-10-02_15-12-31/X.pickle', 'r'))
    y = cPickle.load(open('results/MITOSIS-2016-10-02_15-12-31/y.pickle', 'r'))
    # FACENET
#    X = cPickle.load(open('results/MITOSIS-2016-10-02_15-11-31/X.pickle', 'r'))
#    y = cPickle.load(open('results/MITOSIS-2016-10-02_15-11-31/y.pickle', 'r'))

#    X = cPickle.load(open('results/MITOSIS-2016-09-25_12-34-10/X.pickle', 'r'))
#    y = cPickle.load(open('results/MITOSIS-2016-09-25_12-34-10/y.pickle', 'r'))
#    X = cPickle.load(open('results/MITOSIS-2016-09-29_22-25-05/X.pickle', 'r'))
#    y = cPickle.load(open('results/MITOSIS-2016-09-29_22-25-05/y.pickle', 'r'))
#    X = cPickle.load(open('results/MITOSIS-2016-09-04_20-05-15/X.pickle', 'r'))
#    y = cPickle.load(open('results/MITOSIS-2016-09-04_20-05-15/y.pickle', 'r'))

#    X = cPickle.load(open('results/MITOSIS-2016-09-04_17-50-30/X.pickle', 'r'))
#    y = cPickle.load(open('results/MITOSIS-2016-09-04_17-50-30/y.pickle', 'r'))

#    X = cPickle.load(open('results/MITOSIS-2016-09-01_01-52-52/X.pickle', 'r'))
#    y = cPickle.load(open('results/MITOSIS-2016-09-01_01-52-52/y.pickle', 'r'))

#    X = cPickle.load(open('results/MITOSIS-2016-08-30_22-28-48/X.pickle', 'r'))
#    y = cPickle.load(open('results/MITOSIS-2016-08-30_22-28-48/y.pickle', 'r'))
    #X = cPickle.load(open('results/RNA-2016-08-03_00-03-09/X.pickle','r'))
    #y = cPickle.load(open('results/MITOSIS-2016-08-02_15-47-27/y.pickle', 'r')) #open('results/RNA-2016-08-03_00-03-09/y.pickle','r'))

if args.seed != -1:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=args.split, random_state=int(args.seed))
else:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=args.split)

if args.mode == 'mitosis': # classification problem

    if int(args.experiments) == 1:
        print "[EXP > OOB] Performing oob error rate experiment"
        ensemble_clfs = [
            ("RandomForestClassifier, max_features='sqrt'",
                RandomForestClassifier(warm_start=True, oob_score=True,
                                       max_features="sqrt", n_jobs=-1)),
            ("RandomForestClassifier, max_features='log2'",
                RandomForestClassifier(warm_start=True, max_features='log2',
                                       oob_score=True, n_jobs=-1)),
            ("RandomForestClassifier, max_features=None",
                RandomForestClassifier(warm_start=True, max_features=None,
                                       oob_score=True, n_jobs=-1))
        ]

        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
        
        # Range of `n_estimators` values to explore.
        min_estimators = 10
        max_estimators = 500

        print "\t Ranging from ", min_estimators, " to ", max_estimators

        for label, clf in ensemble_clfs:
            for i in tqdm(range(min_estimators, max_estimators + 1)):
                clf.set_params(n_estimators=i)
                clf.fit(X, y)

                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                error_rate[label].append((i, oob_error))

        # Generate the "OOB error rate" vs. "n_estimators" plot.
        for label, clf_err in error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(mydir, 'mitosis_oob_error.png'))

        #print("[EXP > CALIB] Performing multiclass calibration experiment")

    """
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

        if int(args.pickle) == 1:
            with open(os.path.join(mydir, 'svm-' + score + '.pickle'), 'wb') as f:
                cPickle.dump(clf, f)

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
        print "True: ", y_true
        print "Pred: ", y_pred
        #print(confusion_matrix(y_true, y_pred, labels=[1,2,3]))

    """
    print "[PREDICTION > RF] Tuning via Grid Search"
    
    tuned_parameters = {"max_depth": [10, None],
              "min_samples_split": [2, 10],
              "min_samples_leaf": [1, 10],
              "n_estimators": [10, 60, 80, 100, 120, 140, 160],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(RandomForestClassifier(n_estimators=50), tuned_parameters, cv=5,
                       scoring='%s' % score)
        clf.fit(X_train, y_train)

        if int(args.pickle) == 1:
            with open(os.path.join(mydir, 'rf-' + score + '.pickle'), 'wb') as f:
                cPickle.dump(clf, f)

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
        #print(confusion_matrix(y_true, y_pred, labels=[1,2,3]))
else:
    if int(args.experiments) == 1:
        print "[EXP > OOB] Performing oob error rate experiment"
        ensemble_clfs = [
            ("RandomForestRegressor, max_features='sqrt'",
                RandomForestRegressor(warm_start=True, oob_score=True,
                                       max_features="sqrt", n_jobs=-1)),
            ("RandomForestRegressor, max_features='log2'",
                RandomForestRegressor(warm_start=True, max_features='log2',
                                       oob_score=True, n_jobs=-1)),
            ("RandomForestRegressor, max_features=None",
                RandomForestRegressor(warm_start=True, max_features=None,
                                       oob_score=True, n_jobs=-1))
        ]

        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

        # Range of `n_estimators` values to explore.
        min_estimators = 10
        max_estimators = 500

        print "\t Ranging from ", min_estimators, " to ", max_estimators

        for label, clf in ensemble_clfs:
            for i in tqdm(range(min_estimators, max_estimators + 1)):
                clf.set_params(n_estimators=i)
                clf.fit(X, y)

                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                pred = clf.oob_prediction_
                rho, _ = spearmanr(pred, y)
                error_rate[label].append((i, rho))

        # Generate the "OOB error rate" vs. "n_estimators" plot.
        for label, clf_err in error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(mydir, 'rna_oob_error.png'))

    # rna!
    regressor = RandomForestRegressor(n_estimators=440, oob_score=True, n_jobs=-1, verbose=1)
    regressor.fit(X, y)
    #print regressor.oob_prediction_
    print "R^2 is ", regressor.oob_score_
    rho, pval = spearmanr(regressor.oob_prediction_, y)
    print "Spearman's Rho is ", rho, " with p-val ", pval

    if int(args.pickle) == 1:
        with open(os.path.join(mydir, 'rf-regressor.pickle'), 'wb') as f:
            cPickle.dump(regressor, f)

    if int(args.experiments) == 1:
        importances = regressor.feature_importances_
        std = np.std([tree.feature_importances_ for tree in regressor.estimators_],
                 axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("\t %d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), indices)
        plt.xlim([-1, X.shape[1]])
        plt.savefig(os.path.join(mydir, 'rna_feat_imp.png'))

