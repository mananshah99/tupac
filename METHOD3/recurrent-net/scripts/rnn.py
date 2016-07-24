'''
Adaptation of CRNN to our problem -- we want to train on a SET of images (that characterize a WSI)
'''
from __future__ import print_function
from tqdm import tqdm

import numpy as np
import cv2
import random
from keras.datasets import mnist
from keras.models import Sequential
#from keras.initializations import norRemal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta, SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.models import model_from_json
#import json
from keras.utils import np_utils
from keras import callbacks
remote = callbacks.RemoteMonitor(root='http://10.35.73.88:9000')

# for reproducibility
#np.random.seed(2016)  
#random.seed(2016)

#define some run parameters
batch_size      = 32
nb_epochs       = 100000
maxToAdd        = 25 #40
size            = 60

import glob
import sys

## function to get classes for each image

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

groundtruth = read_groundtruth()

### Define a dictionary of class : image number

true_list = {i : [] for i in range(0, 501)}

for row in groundtruth:
    image_number  = int(row[0])
    image_mitosis = int(row[1])-1 #for keras
    image_RNA     = row[2]

    true_list[image_number].append(image_mitosis)

def load_data():
    
    ## STEP 1: GET THE DICTIONARY OF SOFTMAX PATCHES
    
    numbers = {i : [] for i in range(0, 501)}
    mitosis_softmax_patches = glob.glob('/data/dywang/Database/Proliferation/evaluation/mitko-fcn-heatmaps-norm/*.png')
#   print(mitosis_softmax_patches[0]) ==> /data/dywang/Database/Proliferation/evaluation/mitko-fcn-heatmaps-norm/TUPAC-TR-418_level0_x0000079540_y0000062764.png

    print("load_data => Reading all images")
    bar = tqdm(total=len(mitosis_softmax_patches))    
    for patch_name in mitosis_softmax_patches:
        image_name = patch_name.split('/')[-1]
        image_number = int(image_name.split('_')[0].split('-')[-1])
        numbers[image_number].append(cv2.imread(patch_name, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.)
        bar.update(1)

    bar.close()
    
    return numbers

data_dictionary = load_data()

for key in data_dictionary.keys():
    if len(data_dictionary[key]) == 0:
        data_dictionary.pop(key, None)

print("New data dictionary length", len(data_dictionary.keys()))
print("Building model")
#define our time-distributed setup
model = Sequential()
model.add(TimeDistributed(Convolution2D(8, 4, 4, border_mode='valid'), input_shape=(maxToAdd,1,size,size)))
model.add(Activation('relu'))
model.add(TimeDistributed(Convolution2D(16, 3, 3, border_mode='valid')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),border_mode='valid')))
model.add(Activation('relu'))
model.add(TimeDistributed(Convolution2D(8, 3, 3, border_mode='valid')))
model.add(Activation('relu'))
model.add(Reshape((maxToAdd,np.prod(model.output_shape[-3:])))) #this line updated to work with keras 1.0.2
model.add(TimeDistributed(Flatten()))
model.add(Activation('relu'))
model.add(GRU(output_dim=100,return_sequences=True))
model.add(GRU(output_dim=50,return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(output_dim=3, activation='softmax')) #categorical

rmsprop = RMSprop(lr=0.00001)
model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'], class_mode='categorical')
print("Done building model")
#run epochs of sampling data then training

indices = [i for i in data_dictionary.keys()]
random.shuffle(indices)

indices_train = indices[0 : int(0.85 * len(indices))]
indices_test  = indices[int(0.85 * len(indices)) :]

X_valid     = np.zeros((len(indices_test),maxToAdd,1,size,size))
y_valid     = []

for i in range(0, len(indices_test)):
    #initialize a training example of max_num_time_steps,im_size,im_size
    output      = np.zeros((maxToAdd,1,size,size))

    #decide how many MNIST images to put in that tensor
    numToAdd    = maxToAdd #np.ceil(np.random.rand()*maxToAdd)

    #sample that many images from a given class
    index = indices_test[i] 
    images = data_dictionary[index]

    example_indices = np.random.choice(len(images), size=numToAdd)
    example = np.zeros((numToAdd, 60, 60))

    for i, e in enumerate(example_indices):
        example[i,:,:] = images[e]

    output[0:numToAdd,0,:,:] = example
    X_valid[i,:,:,:,:] = output
    y_valid.append(true_list[index][0])

y_valid     = np.array(y_valid)
y_valid     = np_utils.to_categorical(y_valid, 3)

for ep in range(0,nb_epochs):
    X_train       = []
    y_train       = []
    
    X_train     = np.zeros((len(indices_train),maxToAdd,1,size,size))

    for i in range(0, len(indices_train)): # examplesPer):
        #initialize a training example of max_num_time_steps,im_size,im_size
        output      = np.zeros((maxToAdd,1,size,size))
    
        #decide how many MNIST images to put in that tensor
        numToAdd    = maxToAdd #np.ceil(np.random.rand()*maxToAdd)

        #sample that many images from a given class
        index = indices_train[i]
        images = data_dictionary[index]

        example_indices = np.random.choice(len(images), size=numToAdd)
        example = np.zeros((numToAdd, 60, 60))

        for i, e in enumerate(example_indices):
            example[i,:,:] = images[e]
        #print(example.shape)
        
        output[0:numToAdd,0,:,:] = example
        X_train[i,:,:,:,:] = output
        y_train.append(true_list[index][0])

    y_train     = np.array(y_train)
    y_train     = np_utils.to_categorical(y_train, 3)
    # --- for validation

    if ep == 0:
        print("X_train shape: ",X_train.shape)
        print("y_train shape: ",y_train.shape)
        print("X_valid shape: ",X_valid.shape)
        print("y_valid shape: ",y_valid.shape)
        print(y_train)

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1, validation_data=(X_valid, y_valid),
              verbose=1, callbacks=[remote])

    #classes = model.predict_classes(X_valid, batch_size=64)
    #print(classes)

    jsonstring  = model.to_json()
    with open("../models/rnn-epoch-" + str(ep) + ".json", 'wb') as f:
        f.write(jsonstring)
    model.save_weights("../models/rnn-epoch-" + str(ep) + ".h5",overwrite=True)
