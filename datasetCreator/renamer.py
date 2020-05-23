import glob
import os

path2 = glob.glob('dataset/copyBefore/*')

for i in range(0, len(path2)):
    os.rename(path2[i], "dataset/copyBefore/" + str(i) + ".jpg")