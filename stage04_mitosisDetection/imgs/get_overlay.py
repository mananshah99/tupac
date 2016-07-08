import skimage as ski
import skimage.io as skio
import skimage.measure as skim
import numpy as np
import csv
import cv2

def get_overlay_image(img, t_msk, ground_pts):
    cts = skim.find_contours(t_msk, 0)

    color = [0, 255, 0]
    for ct in cts:
        ct = ct[:, [1, 0]]
        pts = ct.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, 5)

    color2 = [0, 0, 255]
    for pt in ground_pts:
        print "@@", pt
        cv2.circle(img, (pt[1], pt[0]), 10, color2, -1)
    return img

def load_pts(csv_file):
    pts = []
    with open(csv_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            pts.append((int(row[0]), int(row[1])))
    return pts
img = cv2.imread('01.tif')
t_msk = skio.imread('01_mask_nuclei.png')

pts = load_pts('01.csv')

img3 = get_overlay_image(img, t_msk, pts)
cv2.imwrite('a.png', img3)
