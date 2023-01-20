import sys
import os
import shutil
import random

imfolder = sys.argv[1]
test_ratio = float(sys.argv[2])

labels = os.listdir(imfolder)
os.mkdir(os.path.join(imfolder,"test"))
os.mkdir(os.path.join(imfolder,"train"))
for label in labels:
    os.mkdir(os.path.join(imfolder,"test",label))
    os.mkdir(os.path.join(imfolder,"train",label))

for label in labels:
    imlist = os.listdir(os.path.join(imfolder,label))
    imnum = len(imlist)
    numtest = int(imnum*test_ratio)
    testlist = random.sample(imlist,numtest)
    for image in imlist:
        if image in testlist:
            shutil.copy(os.path.join(imfolder,label,image),os.path.join(imfolder,"test",label,image))
            print("copied file ", image)
        else:
            shutil.copy(os.path.join(imfolder,label,image),os.path.join(imfolder,"train",label,image))
            print("copied file", image)
