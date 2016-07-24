Y = "[1 2 1 3 2 1 3 2 1 3 2 1 1 3 1 1 1 1 1 3 3 1 3 1 2 3 1 3 3 3 1 2 2 3 2 3 2 2 1 2 1 1 3 3 1 1 3 1 1 3 2 2 1 3 1 2 3 3 1 3]"
X = "[1 2 1 1 3 1 1 2 3 3 2 3 3 3 3 2 1 3 2 3 1 2 2 1 1 3 1 1 1 3 1 2 3 3 3 2 2 1 3 2 3 1 3 3 3 3 3 1 3 1 2 3 1 3 2 2 3 3 1 3]"

from ast import literal_eval
import re
import numpy as np

def convert(s):
    s = re.sub('\s+', ',', s)
    s = np.array(literal_eval(s))
    return s

X = convert(X)
Y = convert(Y)

print X
print Y

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(Y,X,pos_label=3)
roc_auc = auc(fpr, tpr)

print roc_auc
