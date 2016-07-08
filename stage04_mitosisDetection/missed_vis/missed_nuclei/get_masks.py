# This script obtains nuclei masks and places them in the right format

import os 
import cv2

files = [f for f in os.listdir('.') if os.path.isfile(f) and 'mask' not in f]

prefix = '/data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data'


for f in files:
    outerfolder = f.split('-')[0]
    imname = f.split('-')[1]
    
    fullname = prefix + '/' + outerfolder + '/' + imname[:-4] + '_mask_nuclei.png'

    print (f, fullname)
    
    im = cv2.imread(fullname)
    cv2.imwrite(f + '_mask_nuclei.png',im)
         
