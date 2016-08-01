from ColorNormalization import ColorNormalization

cn = ColorNormalization('/data/dywang/Database/Proliferation/data/TrainingData/training_image_data/TUPAC-TR-001.svs',
                        '/data/dywang/Database/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16/thresholded-0.65/TUPAC-TR-001.png',
                        verbosity=1) 

import functools
import multiprocessing

pool = multiprocessing.Pool(20)

coords = cn.patch_coordinates
mmap = cn.outmap

def process_patch(coordinate): #coordinate is a tuple
    patch = cn.get_image(cn.wsi, coordinate[0], coordinate[1], 0, 1000)
    patch = cn.convert(patch)
    #cv2.imwrite('testdir/converted-' + str(coordinate[0]) + '=' + str(coordinate[1]) + '.png', patch[:,:,(2,1,0)])
    try:
        mmap[coordinate[0] : coordinate[0] + patch.shape[0], coordinate[1] : coordinate[1] + patch.shape[1]] = patch
    except:
        pass #forget it for now

    print "\t process_patch : wrote coordinates ", coordinate

print ">> length of coords is ", len(coords)
pool.map(process_patch, coords)
