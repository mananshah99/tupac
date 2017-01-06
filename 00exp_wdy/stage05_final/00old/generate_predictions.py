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
parser.add_argument('-n', '--name', type=str, required=False, default="predictions.csv", help="Filename for predictions to be stored")

# Set up dictionaries and select image IDs for sampling
args = parser.parse_args()
classifiers = ['svm', 'rf']

## Get names of images

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

image_ids = []
for key in class_dictionary:
    tmp =  class_dictionary[key]
    image_ids.extend(tmp)

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

X_mitosis = cPickle.load(open(join(args.mitosis_directory, 'X.pickle')))
y_mitosis = cPickle.load(open(join(args.mitosis_directory, 'y.pickle')))

X_rna = cPickle.load(open(join(args.rna_directory, 'X.pickle')))
y_rna = cPickle.load(open(join(args.rna_directory, 'y.pickle')))

# mitosis

classifier_list = ['id'] #the headers for the CSV file
mitosis_image_preds = {i : [str(i).zfill(3)] for i in range(1, 501)} #prediction values (each header should have one value in image_preds)

for idx, i in enumerate(image_ids):
    mitosis_image_preds[int(i)].append(y_mitosis[idx])

classifier_list.append("score")

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
        kf = KFold(len(X_mitosis), n_folds=5, shuffle=True) 

        for k, (train, test) in enumerate(kf):
            clf.fit(X_mitosis[train], y_mitosis[train])
            preds = clf.predict(X_mitosis[test])
            for idx, i in enumerate(image_ids[test]):
                mitosis_image_preds[int(i)].append(preds[idx])

        classifier_list.append(f.split('/')[-1].split('.')[0])

write_csv_preds(classifier_list, mitosis_image_preds, 'mitos-predictions.csv')

classifier_list = ['id'] #the headers for the CSV file
rna_image_preds = {i : [str(i).zfill(3)] for i in range(1, 501)} #prediction values (each header should have one value in image_preds)

for idx, i in enumerate(image_ids):
    rna_image_preds[int(i)].append(y_rna[idx])

classifier_list.append("score")

for f in glob.glob(join(args.rna_directory, '*.pickle')):
    if is_regressor(f):
        print "Using regressor ", f.split('/')[-1]

        clf = cPickle.load(open(f, 'r'))        
        clf.set_params(n_jobs=-1, verbose=1)
        kf = KFold(len(X_rna), n_folds=5, shuffle=True) 

        for k, (train, test) in enumerate(kf):
            clf.fit(X_rna[train], y_rna[train])
            preds = clf.predict(X_rna[test])
            for idx, i in enumerate(image_ids[test]):
                rna_image_preds[int(i)].append(preds[idx])

        classifier_list.append(f.split('/')[-1].split('.')[0])

write_csv_preds(classifier_list, rna_image_preds, 'rna-predictions.csv')
