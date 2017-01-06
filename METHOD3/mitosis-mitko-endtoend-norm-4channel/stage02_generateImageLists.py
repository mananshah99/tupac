# image name looks like TUPAC-TR-499_level0_x0000020388_y0000038540-1.png
# let's get the first segment (before '-') and split those, and then keep the 0s and 1s together
import os, sys
from glob import glob
from random import shuffle

# output list looks like this:
#   1000x1000 patch (as many as possible per image) [space] rna or mitosis class
#

train_f = 'train.lst'
test_f = 'test.lst'
train_f = open(train_f, 'wb+')
test_f = open(test_f, 'wb+')

def read_groundtruth(filename = '/data/dywang/Database/Proliferation/libs/METHOD3/training_ground_truth.csv'):
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

MODE = 'mitosis' # or 'rna'

## CREATE MAP

groundtruth = read_groundtruth()
groundtruth_map = {}

for x in groundtruth:
    if MODE == 'mitosis':
        groundtruth_map[x[0]] = str(int(x[1]) - 1) # for caffe
    else:
        groundtruth_map[x[0]] = x[2]

images = glob('outputdir/*.png')

overall_names = set()
for i in images:
    i_first = i[0:-6]
    overall_names.add(i_first)

overall_names_l = list(overall_names)
shuffle(overall_names_l)

train_names = overall_names_l[0:int(0.9*len(overall_names_l))]
val_names   = overall_names_l[int(0.9*len(overall_names_l)):]

print len(train_names), len(val_names)

for n in train_names:
    n_0 = n + "-0.png"
    n_1 = n + "-1.png"

    number = n[n.index('TUPAC') + 9 : n.index('TUPAC') + 12]
    output = groundtruth_map[number]

    train_f.write(n_0 + " " + output + "\n")
    train_f.write(n_1 + " " + output + "\n")

train_f.close()

for n in val_names:
    n_0 = n + "-0.png"
    n_1 = n + "-1.png"

    number = n[n.index('TUPAC') + 9 : n.index('TUPAC') + 12]
    output = groundtruth_map[number]

    test_f.write(n_0 + " " + output + "\n")
    test_f.write(n_1 + " " + output + "\n")

test_f.close()

