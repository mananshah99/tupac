import os
# Keep the same positives, use different negatives
# the initial list (with 0 = non ROI and 1 = ROI) is located here:
#   ../../train/ROI_stage01_LEVEL00/imglist_stage01.lst
#

f = open('roi-level0-stage2-initial.lst', 'r')
out = open('roi-level0-stage2-final.lst', 'wb')

# write the positives first
for line in f:
    line = line.strip('\n')
    spl = line.split(" ")
    if spl[1] == '1':
        out.write(line + '\n') 
    else:
       continue

f.close()
out.close()

out = open('roi-level0-stage2-final.lst', 'a+')

# now write the (more confident) negatives
files = [f for f in os.listdir('ROI-Stage2')]

prefix = '/data/dywang/Database/Proliferation/libs/stage02_genROIPatches/ROI-Stage2/'

for f in files:
    out.write(prefix + f + ' ' + '0\n')

out.close()
