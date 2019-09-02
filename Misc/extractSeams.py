import sys, getopt
import os
import cv2
import numpy as np

# python extractSeams.py ./replace/mytestL2RBorder125_sync.png ./replace/1.png

imageSize = 256

def main(args):
    DIRA = args[0]
    DIRB = args[1]
    if not os.path.isdir("./test1"):
        os.mkdir("./test1")

    serial = 0
    imgA = cv2.imread(DIRA)
    imgA = cv2.resize(imgA, (3072, 3072),)
    imgB = cv2.imread(DIRB)
    imgB = cv2.resize(imgB, (3072, 3072),)
    # print(img)
    for j in range(0, 12 - 1):
        for k in range(0, 12 - 1):
            newImgA = imgA[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            newImgB = imgB[j * imageSize + 128: j * imageSize + 384, k * imageSize + 128: k * imageSize + 384, :]
            newImage = np.hstack([newImgA, newImgB])
            cv2.imwrite("./test1/" + str(serial).zfill(4) + ".png", newImage)
            # cv2.imwrite("./new/" + str(serial).zfill(4) + ".png", newImg)
            serial = serial + 1
            
    return 0

if __name__ == "__main__":
    # print(len(sys.argv))
    main(sys.argv[1:])