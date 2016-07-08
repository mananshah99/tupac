import os, sys, csv
import skimage as ski
import skimage.io as skio
import numpy as np
from nuclei_detection import nuclei_detection
from skimage.morphology import disk
import os.path

# not useful as of yet (Jun 27, 2016)
def gen_ground_truth_mask(imgfiles):
    SIZE = 2084
    cnt = 0
    imgs = []
    coords = []
    centroids = []
    for imgID, imgfile in enumerate(imgfiles):
        print imgID, imgfile
        annotfile = imgfile[:-3] + "csv"
        maskfile = imgfile[:-4] + "_mask.png"
        mask_img = np.zeros((SIZE, SIZE));

        csvReader = csv.reader(open(annotfile, 'rb'))
        for row in csvReader:
            for i in range(0, len(row)/2):
                xv, yv = (int(row[2*i]), int(row[2*i+1]))
                mask_img[yv, xv] = 1
        skio.imsave(maskfile, mask_img)
        #break

def gen_nuclei_mask(imgfiles):
    MinPixel = 200
    MaxPixel = 2500
    for imgID, imgfile in enumerate(imgfiles):
        if "mask" in imgfile:
            continue
        if os.path.isfile(imgfile[:-4] + "_mask_nuclei.png"):
            print "Already have " + imgfile
            continue
        print imgID, imgfile
        try:
            img = skio.imread(imgfile)
            bw1 = nuclei_detection(img, MinPixel, MaxPixel)
            bw1 = bw1 > 0
            bw1 = ski.morphology.dilation(bw1, disk(5)) > 0
            maskfile = imgfile[:-4] + "_mask_nuclei.png"
            skio.imsave(maskfile, bw1*255)
        except Exception as e:
            print e
            import sys
            sys.exit(0)
  #          import subprocess
 #           os.remove(imgfile)
#            print "!!!!! removed " + imgfile
# imgfiles = [l.strip() for l in os.popen('find /data/dywang/Database/Proliferation/data/mitoses/mitoses_image_data/ -name "*.tif"').readlines()]
imgfiles = [l.strip() for l in os.popen('find /data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/patches_06-29-16 -name "*.png"').readlines()]

gen_nuclei_mask(imgfiles)
