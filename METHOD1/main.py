'''
Executes method one for overall classification

Testing (July 5, 2016): Use 
    10 examples/1
    10 examples/2
    10 examples/3

Generate heatmaps for these images using googlenet, and generate feature vectors.
Make sure this is easily extensible to utilize all images
'''

import mitosis_predict
# import rna_predict

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

def generate_input_list(samples_per_class = 10):
    print '__implement__'


groundtruth = read_groundtruth()

### Define a dictionary of class : image number

mitosis_dictionary = {1 : [], 2 : [], 3 : []}

for row in groundtruth:
    image_number  = row[0]
    image_mitosis = row[1]
    image_RNA     = row[2]
    
    mitosis_dictionary[int(image_mitosis)].append(image_number)

### Iterate through the dictionary and select samples, or select all
SAMPLE_SIZE = 10
GEN_HEATMAPS = 1 # if this is set to 0, images that don't have corresponding heatmaps will be ignored

import random
image_ids = []
for key in mitosis_dictionary:
    tmp =  mitosis_dictionary[key] if SAMPLE_SIZE == -1 else random.sample(mitosis_dictionary[key], len(mitosis_dictionary[key]))[0:SAMPLE_SIZE]
    if key == 1:
        tmp = ['001','006','008','009','010','014','015','016','017','018']
    elif key == 2:
        tmp = ['003','004','005','011','013','021','024','026','027','032']
    elif key == 3:
        tmp = ['007','012','019','023','029','030','036','041','046','047']
    image_ids.extend(tmp)

for image_id in image_ids:
    import glob
    globname = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/patches_06-29-16/TUPAC-TR-' + image_id + '*'
    patches = []
    for patch_name in glob.glob(globname):
        patches.append(patch_name)
    
    features = mitosis_predict.extract_features(patches)
