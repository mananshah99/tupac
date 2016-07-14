import os, sys
from os import listdir
from os.path import isfile, join

total_list = []

files = [f for f in listdir('pos_result_maps') if isfile(join('pos_result_maps', f))]

for f in files:
    idx = f.index('_class')
    name = f[0:idx]
    value = int(f[idx+7:-4])

    full_path = '/data/dywang/Database/Proliferation/libs/METHOD3/pos_result_maps/' + name + '_class=' + str(value) + '.png'
    
    string = full_path + ' ' + str(value) + '\n'
    total_list.append(string)

from random import shuffle

shuffle(total_list)

train_list = total_list[0:int (.9 * len(total_list))]
val_list = total_list[int (.9 * len(total_list)) -10 :]

train_file = open('lists/pos_result_map_train.lst', 'wb+')
val_file = open('lists/pos_result_map_val.lst', 'wb+')
for F in train_list:
    train_file.write(F)
train_file.close()
for F in val_list:
    val_file.write(F)
val_file.close()

