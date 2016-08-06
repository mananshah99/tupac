#! /bin/sh
#
# do_get.sh
# Copyright (C) 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
#

sh /home/manan/bin/n rsync -P --rsh=ssh --delete -uar results/mitosis-full-stage2_07-12-16 results/mitosis-train_07-07-16 results/mitosis-train-stage2_07-12-16 ms831@transfer.orchestra.med.harvard.edu:/home/ms831/tupac/stage03_deepFeatMaps/results/
