# 1) get all patch files

PATH = "../../results/patches_06-29-16"

from os import listdir
from os.path import isfile, join
import pprint

onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
onlyfiles = [f for f in onlyfiles if not 'nuclei' in f]

# 2) save them to the right file with the right extension
# should look like
#    ../../../libs/stage03_deepFeatMaps/results/patches_06-27-16/TUPAC-TR-020-(66164,28060).png
#

prefix = '../../../libs/stage03_deepFeatMaps/results/patches_06-29-16/'
outfile = open('../mitosis-updated-withnuclei.lst', 'w+')

onlyfiles.sort(key = lambda x: int(x[9:12]))

N_PATCHES_PER = 15

mp = {x : [] for x in range(1, 500)}

for filename in onlyfiles:
    fID = int(filename[9:12])
    mp[fID].append(filename)

# pprint.pprint(mp)

import random
from scipy.misc import imread

for key in mp:
    if len(mp[key]) == 0:
        continue

    paths = mp[key]
    random.shuffle(paths)
    selection = paths[0:N_PATCHES_PER]

    for selected_path in selection:
        toprint = prefix + selected_path
        #try:
        #    tmp = imread('../' + toprint)
        #except Exception as e:
        #    print e
        #    print selected_path
        #    continue
        outfile.write(toprint + ' ' + toprint[:-4] + '_mask_nuclei.png' + '\n')
    print key
outfile.close()
