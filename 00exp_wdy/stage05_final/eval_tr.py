import os, sys
from extract_wsi import extract_features

import cPickle
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import ml_metrics as metrics
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.grid_search import GridSearchCV

tr_hp = '/home/dayong/ServerDrive/GPU/data/Database/Proliferation/libs/00exp_wdy/stage02_getHP/result_wsi_fcnn_t05/heatmap'
tr_mk = '/home/dayong/ServerDrive/GPU/data/Database/Proliferation/libs/00exp_wdy/stage02_getHP/wsi_msk'

te_hp = ''

TRNUM = 500
TENUM = 321

if 1:
    X_tr = []
    for i in range(1, TRNUM+1):
        hep_path = '%s/TUPAC-TR-%03d.png'%(tr_hp, i)
        msk_path = '%s/TUPAC-TR-%03d.png'%(tr_mk, i)
        print hep_path
        print msk_path
        fv = extract_features(hep_path, msk_path, tv=0.5, ratio=1.0, dscale=0.0001)
        X_tr.append(fv)
    X_tr_array= np.array(X_tr)
    np.save('X_tr', X_tr_array)

else:
    X_tr = np.load('X_tr.npy')
y_tr = [int(l.strip().split(',')[0]) for l in open('/home/dayong/ServerDrive/GPU/data/Database/Proliferation/data/training_ground_truth.csv')]

kappa_scorer = make_scorer(cohen_kappa_score)
parameters = {'n_estimators': [10, 50, 100]}
grid = GridSearchCV(ExtraTreesClassifier(), param_grid = parameters, scoring=kappa_scorer)
grid.fit(X_tr, y_tr)
print grid.best_score_