# TUPAC 2016 Tumor Proliferation Assessment Grand Challenge

This repository implements a complete pipeline to grade the severity of tumors in unlabeled whole slide images as part of the TUPAC 2016 challenge. 

## Pipeline-based Approach

### Stage 1: Whole Slide Image Preprocessing

Code for this stage is housed in the directory `stage01_findTissueRegions`. We reduce each whole slide image to level 2 (a 16x reduction in the size) and perform tissue extraction via otsu's method. 

### Stage 2: ROI (Tumor Region) Extraction 

Code for this stage is housed in the directory `stage02_genROIPatches` (with fully convolutional models in `stage06_fcnSegmentation`), with result feature maps in `stage03_deepFeatMaps`. Two primary steps are conducted here; we first train a CNN-based ROI/non ROI detector (Googlenet) with positives as the ground truth samples and negatives as all external regions (each training image is not thoroughly annotated with positive and negative samples). We subsequently annotate each ground truth image with our detector and remove false positives in our "Stage 2" model, yielding an effective detector that localizes dense tissue. Models developed include
* GoogleNet (Stages 1 and 2 for Level 0, Stage 1 for Level 2)
* FaceNet (Stages 1 and 2)
* Wide Residual Networks (Stage 1, failed to converge)
* VGGNet (Stage 1, memory limits and convergence issues prevented further development)
* ResNet-18, ResNet-34, ResNet-152 (Stage 1, failed to converge and memory limits for 152)

### Stage 3: High Power Region Extraction

Code for this stage is housed in the directory `stage03_deepFeatMaps`, with the most important file located at `stage03_deepFeatMaps/extract_patches_for_mitosis.py`. Here, we extract 50 "high-powered regions" -- that is, regions that accurately reflect the variance and tumor regions in any given whole slide image. These regions are derived from a thresholded ROI map (at level `t = 0.65`) and are stored in a separate results pach directory (in  `stage03_deepFeatMaps`) for future processing. The extraction algorithm computes the patch at the centroid of the tumor region if only one patch is to be computed (and otherwise computes a random region to extract), weighting larger regions with more patches. 

### Stage 4: Staining Normalization

Code for this region is developed in both python and MATLAB. The MATLAB version is hosted at the TUEImage main repository `https://github.com/tueimage/staining-normalization`, and our python adaptation is stored in `StainingNormalizer`. Currently, the high power regions are processed on Orchestra with the MATLAB code, normalized, and stored in a normalized directory located in `stage03_deepFeatMaps`. We have observed that normalized patchs generally perform much better than regular patches with our mitosis detectors, and are therefore developing all future steps solely on the normalized versions.

### Stage 5: Mitosis High Power Heatmap Generation

Code for this stage is quite spread out, with beginning stages hosted in `stage04_mitosisDetection`, some fully convolutional models hosted in `stage06_fcnSegmentation`, and our current detectors hosted in `00exp_wdy`. Models implemented include
* Googlenet (Stages 1 and 2)
* Mitkonet (Stage 1, but works quite well)
* Facenet (Stages 1 and 2)

### Stage 6: Mitosis Whole Slide Heatmap generation

Code for this stage is primarily run on Orchestra, and hosted in `00exp_wdy`. We tile each image into 1000x1000 patches (and stain normalize each patch), and subsequently use a fully convolutional mitosis detector (Mitkonet currently) to generate output heatmaps from the fully convolutional networks. We upsample these outputs to generate an output for the whole slide image. Parallelizing this taks on Orchestra yields a completion time of approximately one day (overnight). 

### Stage 7: Result Evaluation -- Method 1(a)

### Stage 7: Result Evaluation -- Method 1(b)

### Stage 7: Result Evaluation -- Method 2

## End-to-end Deep Learning Approach
