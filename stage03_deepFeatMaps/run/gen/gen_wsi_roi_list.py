import os, sys
from os import listdir
from os.path import join, isfile

bad = [2,45,91,112,205,242,256,280,313,329,467]

def files(PATH):
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    return onlyfiles

original_dir = '/data/dywang/Database/Proliferation/data/TrainingData/training_image_data'
original_files = files(original_dir)

roi_thresh_dir = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/thresholded-0.65'

original_files = [i for i in original_files if (int(i[9:12]) not in bad)]
original_files = sorted(original_files, key=lambda x: int(x[9:12]))

print len(original_files)
output_dir = 'wsi-full.lst'
output_f = open(output_dir, 'wb+')

for f in original_files:
    path_1 = join(original_dir, f)
    path_2 = join(roi_thresh_dir, f[:-4] + '.png')

    output_f.write('../../../../../../../' + path_1 + ' ' + '../../../../../../../' + path_2 + '\n')

output_f.close()
