import os
import numpy as np
open("train_ucf11.list","w")
open("test_ucf11.list","w")
trainfile = open("train_ucf11.list","w+")
testfile = open("test_ucf11.list","w+")
#labels= [7,10,25,32,41,83,88,91,93,96,97]
def convert_images_to_list(rootDir):
    index = -1
    list_dirs = os.listdir(rootDir)
    for dirs in list_dirs:
        sublist = os.listdir(rootDir+"/"+dirs)
        index = index + 1
        #cnt = labels[index]
        for subdirs in sublist:
            innerlist = os.listdir(rootDir+"/"+dirs+"/"+subdirs)
            for innerdirs in innerlist:
                if (os.path.splitext(innerdirs)[1] !=".avi" and os.path.splitext(innerdirs)[1] !=".mpg"):
                    output = rootDir[1:]+"/"+dirs+"/"+subdirs+"/"+innerdirs+" %d"%index
                    if np.random.randint(1, 4, 1) == 1:  # 四分之一的概率等于1，输出到test数据集中
                        print(output,file=testfile)
                    else:
                        print(output,file=trainfile)

convert_images_to_list("../videoFile/ucf-11")