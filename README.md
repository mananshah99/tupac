# TUPAC 2016 Tumor Proliferation Assessment Grand Challenge

This repository implements a complete pipeline to grade the severity of tumors in unlabeled whole slide images as part of the TUPAC 2016 challenge. 

## Stage 1: Whole Slide Image Preprocessing

Code for this stage is housed in the directory `stage01_findTissueRegions`. We reduce each whole slide image to level 2 (a 16x reduction in the size) and perform tissue extraction via otsu's method. 

## Stage 2: ROI (Tumor Region) Extraction 

Code for this stage is housed in the directory `stage02_genROIPatches`, with result feature maps in `stage03_deepFeatMaps`. Two primary steps are conducted here; we first train a CNN-based ROI/non ROI detector (Googlenet) with positives as the ground truth samples and negatives as all external regions (each training image is not thoroughly annotated with positive and negative samples). We subsequently annotate each ground truth image with our detector and remove false positives in our "Stage 2" model, yielding an effective detector that localizes dense tissue. 

## Stage 3: High Power Region Extraction

Code for this stage is housed in the directory `stage03_deepFeatMaps`, with the most important file located at `stage03_deepFeatMaps/extract_patches_for_mitosis.py`. Here, we extract 50 "high-powered regions" -- that is, regions that accurately reflect the variance and tumor regions in any given whole slide image. These regions are derived from a thresholded ROI map (at level `t = 0.65`) and are stored in a separate results pach directory (in  `stage03_deepFeatMaps`) for future processing. The extraction algorithm computes the patch at the centroid of the tumor region if only one patch is to be computed (and otherwise computes a random region to extract), weighting larger regions with more patches. 

## Stage 4: Mitosis Heatmap Generation

Code for this stage is quite spread out, with beginning stages hosted in `stage04_mitosisDetection`, some fully convolutional models hosted in `stage06_fcnSegmentation`, and our current detectors hosted in `00exp_wdy`. 
