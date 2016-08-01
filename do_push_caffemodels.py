'''DO THIS:
eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa
'''

import os, sys
# get all caffemodel files in directory and copy them to orchestra
from subprocess import call

''' sample 
sh /home/manan/bin/n rsync -P --rsh=ssh --delete -uar ./stage06_fcnSegmentation/roi-vgg-13/models/vggnet_iter_506.caffemodel ms831@transfer.orchestra.med.harvard.edu:/home/ms831/tupac/./stage06_fcnSegmentation/roi-vgg-13/models/vggnet_iter_506.caffemodel
'''

def do_command(s):
    BEFORE = "rsync --recursive emptydir/ ms831@transfer.orchestra.med.harvard.edu:/home/ms831/tupac/%s"%(s[2:s.rfind('/')])
    COMMAND = "rsync -P --rsh=ssh --delete -uar %s ms831@transfer.orchestra.med.harvard.edu:/home/ms831/tupac/%s "%(s, s[2:])
    print "\t" + BEFORE
    call(BEFORE, shell=True)
    print "\t" + COMMAND
    call(COMMAND, shell=True)

found = 0
list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk('.'):
    for filename in filenames:
        if filename.endswith('.solverstate'): 
            print "=> Command Running"
            do_command(os.sep.join([dirpath, filename]))
