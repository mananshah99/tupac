import os, sys
import glob
from random import shuffle

# output list looks like this:
#   1000x1000 patch (as many as possible per image) [space] rna or mitosis class
#

def read_groundtruth(filename = 'training_ground_truth.csv'):
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
        groundtruth_map[x[0]] = x[1]
    else:
        groundtruth_map[x[0]] = x[2]

train_lst_name = MODE + '_train.lst'
val_lst_name = MODE + '_val.lst'

train_lst = open(train_lst_name, 'wb+')
val_lst = open(val_lst_name, 'wb+')

prefix = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/patches_07-14-16/'

patches = glob.glob(prefix + '*).png')

lines = []
from tqdm import tqdm
bar = tqdm(total=len(patches))

for patch in patches:
    number = patch[patch.index('TUPAC') + 9 : patch.index('TUPAC') + 12]
    output = groundtruth_map[number]
    lines.append(patch + ' ' + str(int(output)-1) + '\n')
    bar.update(1)
bar.close()

shuffle(lines)

lines_tr = lines[0:int(0.9 * len(lines))]
lines_val = lines[int(0.9 * len(lines)):]

print "Training length is ", len(lines_tr), " and validation length is ", len(lines_val)

for line in lines_tr:
    train_lst.write(line)
train_lst.close()

for line in lines_val:
    val_lst.write(line)
val_lst.close()
