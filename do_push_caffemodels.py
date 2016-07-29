import os, sys
# get all caffemodel files in directory and copy them to orchestra
from subprocess import call

def do_command(s, q):
    COMMAND = "sh /home/manan/bin/n rsync -P --rsh=ssh --delete -uar %s ms831@transfer.orchestra.med.harvard.edu:/home/ms831/tupac/%s"%(s, q)
    print COMMAND
    #call(COMMAND)


