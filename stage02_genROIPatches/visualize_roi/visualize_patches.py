import cv2
import glob
import sys
from skimage import measure
from tqdm import tqdm

def get_images_in_dir(directory):
    return glob.glob(directory + "*.png")

# directory where original images are
original_image_dir = '/data/dywang/Database/Proliferation/data/TrainingData/small_images-level-2/'
patch_image_dir = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/patches_07-14-16/'
# directory where the ROI masks are
mask_image_dir = '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/'

images = get_images_in_dir(original_image_dir)

bar = tqdm(total=len(images))

for image in images:
    # these are all full paths
    original = cv2.imread(image)

    mask_image     = cv2.imread(mask_image_dir + image.split('/')[-1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    _, mask_image   = cv2.threshold(mask_image, int(0.65 * 255), 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original, contours, -1, (0,0,255), 3)

    # for each ROI location, get height and width
    roi_list = glob.glob(patch_image_dir + image.split('/')[-1][:-4] + '*).png')
    
    for roi in roi_list:
        tmp = roi.split('(')[1].split(')')[0].split(',')
        h1_level0 = int(tmp[0])
        w1_level0 = int(tmp[1])
        
        h1_level2 = int(float(h1_level0)/16)
        w1_level2 = int(float(w1_level0)/16)
        
        '''
        Assuming (0,0) is @ top left

        x1,y1 ------
        |          |
        |          |
        |          |
        --------x2,y2
        '''
        cv2.rectangle(original, (w1_level2, h1_level2), (w1_level2 + 63, h1_level2 + 63), (0,255, 0), 5)
        
    cv2.imwrite('results/' + image.split('/')[-1], original)
    bar.update(1)
bar.close()
