import sys, getopt
import os
import cv2
import numpy as np

# python replace.py ./replace/mytestL2RBorder125_sync.png ./results/replace/test1_sync/images ./replace/mytestL2RBorder125_sync_r.png

imageSize = 256
size = 3072

def main(args):
    DIR1 = args[0]
    DIR2 = args[1]
    DIR3 = args[2]
    m = int(size / imageSize)
    img = cv2.imread(DIR1)
    img = cv2.resize(img,(size, size),)
    # print(img)

    images = [name for name in os.listdir(DIR2) if os.path.isfile(os.path.join(DIR2, name)) and ".png" in name and "random_sample01" in name]
    images.sort()
    imagesNumber = len(images)
    print(imagesNumber)

    serial = 0
    for j in range(0, m - 1):
        for k in range(0, m - 1):
            repFileName = DIR2 + images[serial]
            repImg = cv2.imread(repFileName)
            img[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384] = repImg
            serial = serial + 1
    cv2.imwrite(DIR3, img)

if __name__ == "__main__":
    # print(len(sys.argv))
    main(sys.argv[1:])