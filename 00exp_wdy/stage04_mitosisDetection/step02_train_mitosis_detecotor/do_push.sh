#! /bin/sh
#
# do_get.sh
# Copyright (C) 2016 Dayong Wang <dayong.wangts@gmail.com>
#
# Distributed under terms of the MIT license.
#

rsync -P --rsh=ssh --delete -ua . dw140@orchestra.med.harvard.edu:/home/dw140/Proliferation/libs/00exp_wdy/stage04_mitosisDetection/step02_train_mitosis_detecotor/
