import sys, os, csv
DATAROOT = '../../data/mitoses'
#ROOT = 'old.training_examples'
#OROOT = 'old.training_examples_visual'
ROOT = 'training_examples'
OROOT = 'training_examples_visual'

#fdlist = ['pos', 'neg', 'pos_stage2', 'neg_stage2']
import cv2
#fdlist = ['pos_stage2']

def getFolders(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    return fpath

def load_csv(csvf):
    csvReader = csv.reader(open(csvf, 'rb'))
    xylist = []
    for row in csvReader:
        for i in range(0, len(row)/2):
            rv, cv = (int(row[2*i]), int(row[2*i+1]))
            xylist.append((cv, rv))
    return xylist

def drawPTS(img, pts, size, color, fill):
    for pt in pts:
        cv2.circle(img, pt, size, color, fill)
    return img

# draw ground truth
if 1:
    fd = 'pos'
    wsi_names = [l.strip() for l in os.popen('ls %s/%s'%(ROOT, fd))]
    for wsi_name in wsi_names:
        print 'wsi_name: ', wsi_name
        hpf_names = [l.strip() for l in os.popen('ls %s/%s/%s'%(ROOT, fd, wsi_name))]
        for hpf_name in hpf_names:
            print "\thpf_name: ", hpf_name
            hpf_img = '%s/mitoses_image_data/%s/%s.tif'%(DATAROOT, wsi_name, hpf_name)
            msk_img = '%s/mitoses_image_data/%s/%s_mask_nuclei.png'%(DATAROOT, wsi_name, hpf_name)
            hpf_csv = '%s/mitoses_ground_truth/%s/%s.csv'%(DATAROOT, wsi_name, hpf_name)
            gt_pts = load_csv(hpf_csv)

            # drow ground truth mitosis
            img = cv2.imread(hpf_img)
            drawPTS(img, gt_pts, 20, (0, 255, 0), 3)

            # drow nuclei position
            msk_image = cv2.imread(msk_img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            cnts, _ = cv2.findContours(msk_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0,255,255), 2)

            outputFolder = getFolders('%s/%s'%(OROOT, wsi_name))
            cv2.imwrite('%s/%s.jpg'%(outputFolder, hpf_name), img)
            #break
        #break
if 1:
    params = [
    ['pos', 5, (0, 0, 255), 2],
    ['neg', 10, (255, 0, 0), -1]
    ]
    for param in params:
        fd = param[0]
        wsi_names = [l.strip() for l in os.popen('ls %s/%s'%(ROOT, fd))]
        for wsi_name in wsi_names:
            print 'wsi_name: ', wsi_name
            hpf_names = [l.strip() for l in os.popen('ls %s/%s/%s'%(ROOT, fd, wsi_name))]
            for hpf_name in hpf_names:
                print "\t", 'hpf_name: ', hpf_name
                hpf_img = '%s/mitoses_image_data/%s/%s.tif'%(DATAROOT, wsi_name, hpf_name)
                hpf_csv = '%s/mitoses_ground_truth/%s/%s.csv'%(DATAROOT, wsi_name, hpf_name)
                gt_pts = load_csv(hpf_csv)

                outputFolder = getFolders('%s/%s'%(OROOT, wsi_name))
                img = cv2.imread('%s/%s.jpg'%(outputFolder, hpf_name))

                img_names = [l.strip() for l in os.popen('ls %s/%s/%s/%s'%(ROOT, fd, wsi_name, hpf_name))]
                pts = []
                for img_name in img_names:
                    itms = img_name.split('.')[0].split('_')
                    #print itms
                    x = int(itms[2][1:])
                    y = int(itms[3][1:])
                    pts.append((x, y))
                drawPTS(img, pts, *param[1:])
                cv2.imwrite('%s/%s.jpg'%(outputFolder, hpf_name), img)
                #break
            #break
        #break
