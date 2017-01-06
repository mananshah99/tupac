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

mydir = os.path.join(os.getcwd(), "testresults/" + args.mode.upper() + "-" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

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

image_ids = []
for i in range(1, 322):
    si = str(i).zfill(3)
    image_ids.append(si)

print image_ids
print "Using ", len(image_ids), " ids, with SAMPLE_SIZE set to ", args.portion

X = []
y = []

def process_images_nopatch(image_id):
    global args
    
    heatmap_dictionary = args.feature_directory
    name = heatmap_directory + '/TUPAC-TE-' + image_id + '.png'
    features = extract_wsi.extract_features(name)

    print "\t Completed ", image_id
    return np.array(features, dtype=np.float32)
#    return preprocessing.scale(np.array(features, dtype=np.float32))

if args.load == 0:
    # If we are looking at patches, process the patches
    if args.ispatch:
        print "fix this later"
        # again, fix this later   
    else:
        heatmap_directory = args.feature_directory
        print "WSI argument selected. Obtaining images from directory ", heatmap_directory
        pool = mp.Pool(processes = 10)
        X = pool.map(process_images_nopatch, image_ids)

    X = np.vstack(X)

    if int(args.pickle) == 1:
        with open(os.path.join(mydir, 'X.pickle'), 'wb') as f:
            cPickle.dump(X, f)
