#!/bin/bash

curdir=`pwd`

cmd="python /home/dywang/WWW/deepzoom/visual_v2.py -l 10.35.73.88 -p 5011 -n ROI \
    --heatmap_threshold 0.7\
    --overlap_ratio 0.5\
    -r /home/dywang/Proliferation/libs/stage03_deepFeatMaps/results/roi-level1_06-24-16\
    --debug \
    /home/dywang/Proliferation/data/TrainingData/small_images-level-3"

echo $cmd
exec $cmd
