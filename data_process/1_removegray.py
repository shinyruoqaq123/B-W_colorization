import os.path
import shutil

import skimage.io as io
import sys

rootdir = sys.argv[1]
folder_name=rootdir.split(os.path.sep)[-1]
newrootdir = rootdir.replace(folder_name,'{}_valid'.format(folder_name))

for parent, dirnames, filenames in os.walk(rootdir):
    i=0
    for filename in filenames:
        i+=1
        if i%100==0:
            print(i)
        path = os.path.join(parent, filename)
        img = io.imread(path)
     
        # remove gray image of 1 channel
        if len(img.shape) == 2:
            newpath = os.path.join(newrootdir, os.path.split(parent)[1])
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            shutil.move(path, newpath)
