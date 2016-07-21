import os,sys
def copy_HPS():
    ROOT='/home/dywang/Proliferation/libs/stage03_deepFeatMaps/results'
    ochestra='transfer.orchestra.med.harvard.edu'
    for line in [l.strip() for l in open('hpr_list.lst')]:
        itms = line.split('/')[-1][:-4].split('-')
        name_prefix = '-'.join(itms[:-1])
        position = itms[-1]
        print name_prefix, position
        x, y = [int(i.strip()) for i in position[1:-1].split(',')]
        nname = '%s_level0_x%010d_y%010d.png'%(name_prefix, x, y)
    #    cmd = 'rsync -P --rsh=ssh "%s/%s" dw140@%s:/home/dw140/work/Database/Proliferation/libs/00exp_wdy/stage02_getHPR/hps_manan_160714/%s/%s'%(ROOT, line, ochestra, name_prefix, nname)
    #    cmd = 'scp "%s/%s" dw140@%s:/home/dw140/work/Database/Proliferation/libs/00exp_wdy/stage02_getHPR/hps_manan_160714/%s/%s'%(ROOT, line, ochestra, name_prefix, nname)
        cmd = 'cp "%s/%s" hps_manan_160714/%s/%s'%(ROOT, line, name_prefix, nname)
        os.system(cmd)
        print cmd

def copy_HPS_neclei():
    ROOT='/home/dywang/Proliferation/libs/stage03_deepFeatMaps/results'
    ochestra='transfer.orchestra.med.harvard.edu'
    for line in [l.strip() for l in open('hpr_list.lst')]:
        itms = line.split('/')[-1][:-4].split('-')
        name_prefix = '-'.join(itms[:-1])
        position = itms[-1]
        print name_prefix, position
        x, y = [int(i.strip()) for i in position[1:-1].split(',')]
        nname = '%s_level0_x%010d_y%010d_mask_nuclei.png'%(name_prefix, x, y)
        ori_image = '%s/%s_mask_nuclei.png'%(ROOT, line[:-4])
        if not os.path.exists(ori_image):
            print "ERROR:", name_prefix, ori_image
            break
        else:
            cmd = 'cp "%s" hps_manan_160714_nuclei_mask/%s/%s'%(ori_image, name_prefix, nname)
            os.system(cmd)
            
copy_HPS_neclei()
