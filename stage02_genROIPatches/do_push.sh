#! /bin/sh
#
# do_get.sh
# Copyright (C) 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
#

sh /home/manan/bin/n rsync -P --rsh=ssh --delete -uar ROI-Stage2 ms831@transfer.orchestra.med.harvard.edu:/home/ms831/tupac/stage02_genROIPatches/ROI-Stage2 
