This repository implements numerous stages in the pipeline for the TUPAC 2016 Challenge. 

Stage 1: Whole Slide Image Processing
    => Step 1: Extract PNG images of the WSIs at level 2 magnification
    => Step 2: Extract tissue regions within the WSI PNG images to alleviate 
        computational demand later on


Stage 2: Region of Interest Extraction
    => Used to further minimize computational power by finding potential regions of
        interest within tissue regions of WSIs
    => TBD, may not be needed


Stage 3: Tumor & Mitosis Feature Extraction
    => Utilize multiple convolutional models to determine heatmaps of locations of both
        tumors and mitoses within WSI images
    => Contains two unique impmenentations:
        -> A GPU based implementation (that runs only on the Beck lab cluster) which takes
            approximately [1-4]hrs to extract heatmaps from any given image
        -> A GPU + CPU + Orchestra based implementation (in gen_wsi_par_tupac.py) which 
            takes approximately 5 minutes to extract heatmaps from any given image (this
            is currently in development)


Stage 4: Classical Feature Extraction
    => Utilize mitosis and tumor heatmaps to generate a feature vector of relevant information
        that characterizes each image (and can be used for the classification problem in stage 1
        as well as the regression problem in stage 2)
    => Basic idea is to use locations (centroids) of each mitosis and extract small patches around
        the mitosis from which to get positional/etc. features as well as features determined
        directly from the heatmap itself
    => Additionally, follow the rules of the challenge (10 in a row is stage 3, etc.)


Stage 5: Fully Convolutional Feature Extraction
    => Utilize previously developed heatmaps (and pass them through fully convolutional network
        architectures) to make more salient predictions 
