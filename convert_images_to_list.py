import os
import numpy as np
open("./list/train.list","w")
open("./list/test.list","w")
trainfile = open("./list/train.list","w+")
testfile = open("./list/test.list","w+")
def convert_images_to_list(rootDir):
    cnt = -1
    list_dirs = os.listdir(rootDir)
    for dirs in list_dirs:
        sublist = os.listdir(rootDir+"/"+dirs)
        cnt = cnt + 1
        for subdirs in sublist:
            innerlist = os.listdir(rootDir+"/"+dirs+"/"+subdirs)
            for innerdirs in innerlist:
                if (os.path.splitext(innerdirs)[1] !=".avi" and os.path.splitext(innerdirs)[1] !=".mpg"):
                    output = rootDir+"/"+dirs+"/"+subdirs+"/"+innerdirs+" %d"%cnt
                    if np.random.randint(1, 4, 1) == 1:  # 四分之一的概率等于1，输出到test数据集中
                        print(output,file=testfile)
                    else:
                        print(output,file=trainfile)

convert_images_to_list("./videoFile/ucf-11")