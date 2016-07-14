import os
import sys
from random import shuffle

def split_percentage(list_a, percentage):
    shuffle(list_a)
    count = int(len(list_a) * percentage)
    if not count: return []  # edge case, no elements removed
    list_a[-count:], list_b = [], list_a[-count:]
    return list_b, list_a

import argparse
parser = argparse.ArgumentParser(description='Create train and validation lists from generated patches')
parser.add_argument('--balance', dest='balance_classes', action='store_true')
parser.add_argument('--no-balance', dest='balance_classes', action='store_false')
parser.set_defaults(balance_classes=True)

args = vars(parser.parse_args())

balance_classes=args["balance_classes"]

# 90/10 split of mitotic patches
TRAIN_OUT = '/data/dywang/Database/Proliferation/libs/stage04_mitosisDetection/training_examples/train3.lst'
VAL_OUT   = '/data/dywang/Database/Proliferation/libs/stage04_mitosisDetection/training_examples/val3.lst'

# there are very few false negatives so let's use all we have
posfiles = [l.strip() for l in os.popen('find /data/dywang/Database/Proliferation/libs/stage04_mitosisDetection/training_examples/pos* -name "*.png"').readlines()]
# there are so many false positives let's just use neg_stage2
negfiles = [l.strip() for l in os.popen('find /data/dywang/Database/Proliferation/libs/stage04_mitosisDetection/training_examples/neg_stage2 -name "*.png"').readlines()]

if balance_classes:
    #negfiles, _ = split_percentage(negfiles, float(len(posfiles))/len(negfiles))
    shuffle(negfiles) # for this one only
    negfiles = negfiles[0:4*len(posfiles)]
print "INFO: Found " + str(len(posfiles)) + " mitosis patches and " + str(len(negfiles)) + " non-mitotic patches."

train_pos, val_pos = split_percentage(posfiles, .90)
train_neg, val_neg = split_percentage(negfiles, .90)

TRAIN_OUT_F = open(TRAIN_OUT, 'w+')
VAL_OUT_F   = open(VAL_OUT, 'w+')

for f in train_pos:
    TRAIN_OUT_F.write(f + ' ' + '1\n')

for f in val_pos:
    VAL_OUT_F.write(f + ' ' + '1\n')

for f in train_neg:
    TRAIN_OUT_F.write(f + ' ' + '0\n')

for f in val_neg:
    VAL_OUT_F.write(f + ' ' + '0\n')

TRAIN_OUT_F.close()
VAL_OUT_F.close()
