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

TRAIN_OUT = 'roi-train-L0.lst'
VAL_OUT = 'roi-val-L0.lst'
TEST_OUT = 'roi-test-L0.lst'

negfiles = [l.strip() for l in os.popen('find /data/dzhu1/LEVEL00/sample_W256_P0000000999/img-Normal/ -name "*.png"').readlines()]
posfiles = [l.strip() for l in os.popen('find /data/dzhu1/LEVEL00/sample_W256_P0000000999/img-ROI/ -name "*.png"').readlines()]

if balance_classes:
    if len(negfiles) > len(posfiles):
    	negfiles, _ = split_percentage(negfiles, float(len(posfiles))/len(negfiles))
    else:
        posfiles, _ = split_percentage(posfiles, float(len(negfiles))/len(posfiles))
	
print "INFO: Found " + str(len(posfiles)) + " ROI patches and " + str(len(negfiles)) + " non-ROI patches."

train_val_pos, test_pos = split_percentage(posfiles, .80)
train_val_neg, test_neg = split_percentage(negfiles, .80)

train_pos, val_pos = split_percentage(train_val_pos, .90)
train_neg, val_neg = split_percentage(train_val_neg, .90)

TRAIN_OUT_F = open(TRAIN_OUT, 'w+')
VAL_OUT_F   = open(VAL_OUT, 'w+')
TEST_OUT_F  = open(TEST_OUT, 'w+')

for f in train_pos:
    TRAIN_OUT_F.write(f + ' ' + '1\n')

for f in val_pos:
    VAL_OUT_F.write(f + ' ' + '1\n')

for f in train_neg:
    TRAIN_OUT_F.write(f + ' ' + '0\n')

for f in val_neg:
    VAL_OUT_F.write(f + ' ' + '0\n')

for f in test_pos:
    TEST_OUT_F.write(f + ' ' + '1\n')

for f in test_neg:
    TEST_OUT_F.write(f + ' ' + '0\n')

TEST_OUT_F.close()
TRAIN_OUT_F.close()
VAL_OUT_F.close()

