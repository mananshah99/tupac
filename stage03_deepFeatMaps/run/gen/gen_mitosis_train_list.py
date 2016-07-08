# 1) get all patch files
# oh god this is ugly

PATH = "../../results/patches_06-27-16"

CMD = 'find /data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/ -name "*.tif"'
CMD_2 = 'find /data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/ -name "*.png"'

import os
from os import listdir
from os.path import isfile, join

imagefiles = [l.strip() for l in os.popen(CMD).readlines()]
maskfiles  = [l.strip() for l in os.popen(CMD_2).readlines()]

#  /data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/02/20.tif /data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/02/19_mask_nuclei.png

outfile = open('../mitosis-train.lst', 'w+')

print len(maskfiles), len(imagefiles)

for i in range(0, len(maskfiles)):
    prefix = "../../../../../../../"
    firsthalf = imagefiles[i]
    secondhalf = imagefiles[i][:-4] + "_mask_nuclei.png"
    line = prefix + firsthalf + " " + prefix + secondhalf
    outfile.write(line + '\n')

outfile.close()

