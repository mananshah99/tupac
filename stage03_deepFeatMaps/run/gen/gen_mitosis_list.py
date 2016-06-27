# 1) get all patch files

PATH = "../../results/patches_06-27-16"

from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]

# 2) save them to the right file with the right extension
# should look like
#    ../../../libs/stage03_deepFeatMaps/results/patches_06-27-16/TUPAC-TR-020-(66164,28060).png
#

prefix = '../../../libs/stage03_deepFeatMaps/results/patches_06-27-16/'
outfile = open('../mitosis.lst', 'w+')

for filename in onlyfiles:
    toprint = prefix + filename
    outfile.write(toprint + '\n')

outfile.close()

