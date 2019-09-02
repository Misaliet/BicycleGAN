import sys
import os
import cv2
import numpy as np

# python combineImages.py ./4for1 3072

imageSize = 256

def main(args):
    DIR = args[0]
    size = int(args[1])
    saveName = args[2]
    if size % imageSize != 0:
        print('illegal size!')
        sys.exit()
    m = int(size / imageSize)
    images = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and ".png" in name and "random_sample01" in name]
    images.sort()
    imagesNumber = len(images)
    # print(imagesNumber)
    if imagesNumber % (m * m) != 0:
        print('illegal images number!')
        sys.exit()

    if not os.path.isdir("./whole"):
        os.mkdir("./whole")
    
    n = int(imagesNumber / (m * m))
    l = 0
    for i in range(0, n):
        temp = np.zeros(shape=(size, size, 3))
        n = i * m * m
        for j in range(0, m):
            for k in range(0, m):
                fileName = DIR + "/" + images[l]
                img = cv2.imread(fileName)
                temp[j * imageSize: j * imageSize + imageSize, k * imageSize: k * imageSize + imageSize, :] = img
                l += 1

        # cv2.imwrite("./whole/" + str(i) + ".png", temp)
        cv2.imwrite("./whole/" + saveName, temp)
            
    return 0

if __name__ == "__main__":
    # print(len(sys.argv))
    main(sys.argv[1:])