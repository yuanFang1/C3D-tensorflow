import os
open("label.txt","w")
outfile = open("label.txt","w+")
def getlabel(rootDir):
  cnt = -1
  list_dirs = os.listdir(rootDir)
  for dirs in list_dirs:
    cnt = cnt + 1
    output = dirs+" %d"%cnt
    print(output,file=outfile)
getlabel("../videofile/UCF-101")