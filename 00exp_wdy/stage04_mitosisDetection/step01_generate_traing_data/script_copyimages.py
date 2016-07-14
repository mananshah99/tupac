import sys, os
ROOT = '../stage04_mitosisDetection/training_examples'
#fdlist = ['pos', 'neg', 'pos_stage2', 'neg_stage2']
#fdlist = ['pos_stage2']
fdlist = ['neg']

def getFolders(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)

for fd in fdlist:
    print 'ls %s/%s/*.png'%(ROOT, fd)
    imgs = [l.strip() for l in os.popen('ls %s/%s'%(ROOT, fd)).readlines()]
    for img in imgs:
        img_name = img.split('.')[0]
        itms = img_name.split('-')
        fname = itms[0]
        iname = itms[1]
        x, y = position = itms[2][1:-1].split(',')
        x = int(x)
        y = int(y)
        n_img_name = '%s_%s_%010d_%010d.png'%(fname, iname, x, y)
        outputFolder ='old.training_examples/%s/%s/%s'%(fd, fname, iname)
        getFolders(outputFolder)
        cmd = 'cp "%s/%s/%s" %s/%s'%(ROOT, fd, img, outputFolder, n_img_name)
        print cmd
        os.system(cmd)
    break