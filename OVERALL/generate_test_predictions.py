import os
import cPickle
import sys
import argparse
import glob
import warnings
from os.path import join
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn import *
from sklearn.cross_validation import KFold
from utils import *
import numpy as np

parser = argparse.ArgumentParser(description="Generate cross-validated predictions for TUPAC 2016 given a directory for mitosis classifiers and a directory for RNA regressors")
parser.add_argument('-m', '--mitosis-directory', type=str, required=True, help="Directory where mitosis classifiers are stored (as pickle files)")
parser.add_argument('-r', '--rna-directory', type=str, required=True, help="Directory where RNA classifiers are stored (as pickle files)")

# Set up dictionaries and select image IDs for sampling
args = parser.parse_args()
classifiers = ['svm', 'rf']

## Get names of images

image_ids = []
for i in range(1, 322):
    si = str(i).zfill(3)
    image_ids.append(si)

print image_ids

image_ids = np.array(image_ids)

# image_ids is the ordered list of predictions

def is_classifier(s):
    global classifiers
    
    for i in classifiers:
        if i in s and 'regress' not in s:
            return True
    return False

def is_regressor(s):
    global regressors
    if 'regress' in s:
        return True 
    return False

# for #1: 
#X_mitosis = cPickle.load(open(join('testresults/MITOSIS-2016-08-31_21-08-39/', 'X.pickle')))
# for #2:
#X_mitosis = cPickle.load(open(join('testresults/MITOSIS-2016-10-01_23-03-11/', 'X.pickle')))
# for #3:
#X_mitosis = cPickle.load(open(join('testresults/MITOSIS-2016-10-02_12-00-15/', 'X.pickle')))

# for mitkonet - latest
X_mitosis = cPickle.load(open('10116_FACENET_TEST_X.pickle'))
X_rna = X_mitosis

# mitosis

classifier_list = ['id'] #the headers for the CSV file
mitosis_image_preds = {i : [str(i).zfill(3)] for i in range(1, 322)} #prediction values (each header should have one value in image_preds)

def write_csv_preds(headers, preds_map, out_f):
    out_f = open(out_f, 'wb+')
    header = ",".join(headers)
    out_f.write(header + '\n') 
    
    for key in preds_map:
        pd = preds_map[key]
        line = ",".join(str(v) for v in pd)
        out_f.write(line + '\n')

    out_f.close()

for f in glob.glob(join(args.mitosis_directory, '*.pickle')):
    if is_classifier(f):
        print "Using classifier ", f.split('/')[-1]

        clf = cPickle.load(open(f, 'r'))        
        clf.set_params(n_jobs=-1, verbose=1)
        preds = clf.predict(X_mitosis)
        for idx, i in enumerate(image_ids):
            mitosis_image_preds[int(i)].append(preds[idx])

        classifier_list.append(f.split('/')[-1].split('.')[0])

write_csv_preds(classifier_list, mitosis_image_preds, 'mitos-predictions.csv')

classifier_list = ['id'] #the headers for the CSV file
rna_image_preds = {i : [str(i).zfill(3)] for i in range(1, 322)} #prediction values (each header should have one value in image_preds)

for f in glob.glob(join(args.rna_directory, '*.pickle')):
    if is_regressor(f):
        print "Using regressor ", f.split('/')[-1]

        clf = cPickle.load(open(f, 'r'))        
        clf.set_params(n_jobs=-1, verbose=1)

        preds = clf.predict(X_rna)
        for idx, i in enumerate(image_ids):
            rna_image_preds[int(i)].append(preds[idx])

        classifier_list.append(f.split('/')[-1].split('.')[0])

write_csv_preds(classifier_list, rna_image_preds, 'rna-predictions.csv')
