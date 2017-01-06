import glob

general_directory = "/data/dywang/Database/Proliferation/libs/00exp_wdy/stage04_mitosisDetection/step01_generate_traing_data/training_images_cn/"

pos_directory = general_directory + "pos/*/*/*.jpg"
neg_1_directory = general_directory + "neg/*/*/*.jpg"
neg_2_directory = general_directory + "neg2/*/*/*.jpg"
neg_3_directory = general_directory + "neg3/*/*/*.jpg"

neg_1_files = glob.glob(neg_1_directory)
neg_2_files = glob.glob(neg_2_directory)
neg_3_files = glob.glob(neg_3_directory)

pos_files = glob.glob(pos_directory)

#print "[INFO] There are ", len(neg_1_files) + len(neg_2_files) + len(neg_3_files), " negative files and ", len(pos_files), "positive files"

import numpy as np

np.random.seed(0)

NEG_BIAS = 1 # same number of pos and neg if 1

neg_files = neg_1_files + neg_2_files + neg_3_files
neg_choice = np.random.choice(neg_files, size=len(pos_files)*NEG_BIAS, replace=False)

# now split into training and testing, 0.9 for train and 0.1 for test

import random
random.seed(0)

random.shuffle(neg_choice)
random.shuffle(pos_files)

pos_tr = pos_files[0:int(0.9*len(pos_files))]
pos_te = pos_files[int(0.9 * len(pos_files)):]
neg_tr = neg_choice[0:int(0.9*len(neg_choice))]
neg_te = neg_choice[int(0.9*len(neg_choice)):]

tr_list = open("list_bias-" + str(NEG_BIAS)+"_tr.lst", "a+")
te_list = open("list_bias-" + str(NEG_BIAS)+"_te.lst", "a+")

for image in neg_tr:
    tr_list.write(image + " 0\n")
for image in pos_tr:
    tr_list.write(image + " 1\n")

for image in neg_te:
    te_list.write(image + " 0\n")
for image in pos_te:
    te_list.write(image + " 1\n")

