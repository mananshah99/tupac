import os, sys
import skimage.io as skio

img1 = 'stage01/01/01_Normalized.tif.png'
img2 = 'stage02/mnet_stage02_m8_90K/heatmap/mitoses_image_data_cn3/01/01_Normalized.tif.png'

i1 = skio.imread(img1)
i2 = skio.imread(img2)

i1[i1 < 0.2 * 255] = 0
i2[i2 < 0.2 * 255] = 0

i1_mask = i1>0
i2_mask = i2>0

i = (i1 + i2) / 2.0
i[~(i1_mask & i2_mask)] = 0