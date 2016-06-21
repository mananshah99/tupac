from count_mitoses import *
import tamura

HEATMAP_IMAGE_FILE = '../stage03_deepFeatMaps/results/mitosis_06-21-16/TUPAC-TR-066.png'
HEATMAP_IMAGE = scipy.misc.imread(HEATMAP_IMAGE_FILE)

number, image = count_mitoses(HEATMAP_IMAGE_FILE)
