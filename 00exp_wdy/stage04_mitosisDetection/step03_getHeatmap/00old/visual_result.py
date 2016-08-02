import os, sys
import skimage as ski
import skimage.io as skio
import cv2

ROOT='/home/dayong/ServerDrive/GPU/data/Database/Proliferation/data/mitoses/mitoses_image_data'

lines = [l.strip() for l in open('tmp.lst')]
for line in lines:
    img_name, msk_name = line.strip().split()
    img_path = '%s/%s'%(ROOT, img_name)
    msk_path = '%s/%s'%(ROOT, msk_name)
    hep_path = 'tmp/%s'%(img_name).replace('.tif', '.png')

    img = cv2.imread(img_path)
    msk_image = cv2.imread(msk_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cnts, _ = cv2.findContours(msk_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0,255,255), 2)

#    msk = msk.astype(np.uint8)
    alpha = 0.3
    hep_image = cv2.imread(hep_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    hep_image_jet = cv2.applyColorMap(hep_image, cv2.COLORMAP_JET)
    print img.shape, hep_path
    img_overlay = cv2.addWeighted(img, alpha, hep_image_jet, 1 - alpha, 1);

    cv2.imwrite('aa.png', img_overlay)
