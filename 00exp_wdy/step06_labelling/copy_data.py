import sys, os

imgs = [l.strip() for l in os.popen('ls top2-mitko-pixel-heatmaps-norm/*.png')]
for img in imgs:
    img_name = img.split('/')[-1]
    print img_name
    fd = img_name.split('_')[0]
    print fd
    cmd = 'cp /home/dayong/ServerDrive/Orchestra/Proliferation/libs/00exp_wdy/stage02_getHPR/hps_manan_160714/%s/%s top2-imgs/%s'%(fd, img_name, img_name)
    print cmd
    os.system(cmd)
