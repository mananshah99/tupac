import os
import cv2
from skimage.color import *
from skimage.draw import *
from skimage.io import *
nucleifiles = [l.strip() for l in os.popen('find /data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/ -name "*.png"').readlines()]

# /data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/02/19_mask_nuclei.png

tp = 0
fn = 0
for nuclei_mask in nucleifiles:
    tmp = nuclei_mask[len(nuclei_mask)-21:]
    prefix = tmp.split("/")[0]
    middle = tmp.split("/")[1][0:2]
    
    csvname = '/data/dywang/Database/Proliferation/data/mitoses/mitoses_ground_truth/' + prefix + '/' + middle + '.csv' 
    if os.path.isfile(csvname):
        f = open(csvname)
        nuclei_im = imread(nuclei_mask)
        
        original_image = '/data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/' + prefix + '/' + middle + '.tif'
        original_im = cv2.imread(original_image)
        
        for line in f:
            if ',' in line:
                y = int(line.split(',')[0])
                x = int(line.split(',')[1])
            try:
                if im[y, x] > 0: #white
                    tp += 1
                if 1: #else:
                    # visualize
                    original_image = '/data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/' + prefix + '/' + middle + '.tif'
                    original_im = cv2.imread(original_image)
                    cv2.circle(original_im, (x,y), 40,255,3)
                    cv2.imwrite(prefix + '-' + middle + '.png', original_im)
                    print prefix + '-' + middle + '.png'
                    fn += 1
            except Exception as e:
                print e
                pass
print float(tp)/(tp+fn)
