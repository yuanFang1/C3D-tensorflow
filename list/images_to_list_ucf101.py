import os
import numpy as np
open("train_ucf101.list","w")
open("test_ucf101.list","w")
trainfile = open("train_ucf101.list","w+")
testfile = open("test_ucf101.list","w+")
def convert_images_to_list(rootDir):
    cnt = -1
    list_dirs = os.listdir(rootDir)
    for dirs in list_dirs:
        sublist = os.listdir(rootDir+"/"+dirs)
        cnt = cnt + 1
        for subdirs in sublist:
            if (os.path.splitext(subdirs)[1] !=".avi" and os.path.splitext(subdirs)[1] !=".mpg"):
                output = rootDir[1:]+"/"+dirs+"/"+subdirs+" %d"%cnt
                if np.random.randint(1, 4, 1) == 1:  # 四分之一的概率等于1，输出到test数据集中
                    print(output,file=testfile)
                else:
                    print(output,file=trainfile)

convert_images_to_list("../videoFile/UCF-101")