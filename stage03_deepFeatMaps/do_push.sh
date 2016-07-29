#! /bin/sh
#
# do_get.sh
# Copyright (C) 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
#

sh /home/manan/bin/n rsync -P --rsh=ssh --delete -uar results/patches_07-14-16 results/roi-level1_06-24-16 results/patches_07-14-16-norm results/roi-overlay_06-22-16 ms831@transfer.orchestra.med.harvard.edu:/home/ms831/tupac/stage03_deepFeatMaps/results/
