import sys, getopt
import os
import cv2
import numpy as np

def imageResize(oldPath, width, height):
    # print("111")
    print(oldPath)
    img = cv2.imread(oldPath)
    # print(img.shape)
    img2 = cv2.resize(img,(width, height),)
    return img2

def imageCut(img, size, folder, k, digit):
    length = img.shape[0]
    num = int (length / size)
    for i in range(0, num):
        for j in range(0, num):
            newImg = img[i * size : (i + 1)* size, j * size: (j + 1) * size]
            cv2.imwrite(folder + str(k).zfill(digit) + ".png", newImg)
            k += 1

def imageSplice(img1, img2, newPath):
    newImage = np.hstack([img1, img2])
    cv2.imwrite(newPath, newImage)

def test():
    imgS = imageResize("./se2934.tif", 3072, 3072)
    imgR = imageResize("./se2934_rgb_250_01.jpg", 3072, 3072)
    print("imageResize")

    imageCut(imgS, 256, "./sketch/", 0, 5)
    imageCut(imgR, 256, "./realScene/", 0, 5)
    print("imageCut")
    
    num = 3072 / 256
    num = num * num + 1
    for i in range(1, int(num)):
        newPath = "./images/" + str(i) + ".jpg"
        img1 = cv2.imread("./sketch/" + str(i) + ".jpg")
        img2 = cv2.imread("./realScene/" + str(i) + ".jpg")
        imageSplice(img1, img2, newPath)
    print("imageSplice")

    print("finish!")

def main(argv):
    sketchPath = ""
    realScenePath = ""
    mode = ""
    digit = 5
    try:
        opts, args = getopt.getopt(argv,"hs:r:m:d:",["sketch=","realScene=", "mode=", "digit="])
    except getopt.GetoptError:
        print('imageProcess.py -s <sketchPath> -r <realScenePath> -m <mode> -d <digit>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('imageProcess.py -s <sketchPath> -r <realScenePath> -m <mode> -d <digit>')
            sys.exit()
        elif opt in ("-s", "--sketch"):
            sketchPath = arg
        elif opt in ("-r", "--realScene"):
            realScenePath = arg
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-d", "--digit"):
            digit = int(arg)
    
    imgS = imageResize(sketchPath, 3072, 3072)
    imgR = imageResize(realScenePath, 3072, 3072)
    print("imageResize")


    if not os.path.isdir("./sketch"):
        os.mkdir("./sketch")
    if not os.path.isdir("./realScene"):
        os.mkdir("./realScene")
    if not os.path.isdir("./images"):
        os.mkdir("./images")
    DIR = './images'
    k = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and ".png" in name])

    imageCut(imgS, 256, "./sketch/", k, digit)
    imageCut(imgR, 256, "./realScene/", k, digit)
    print("imageCut")
    
    num = 3072 / 256
    num = num * num
    for i in range(k, k + int(num)):
        newPath = "./images/" + str(i).zfill(digit) + ".png"
        img1 = cv2.imread("./sketch/" + str(i).zfill(digit) + ".png")
        img2 = cv2.imread("./realScene/" + str(i).zfill(digit) + ".png")
        # change img1 and img2 based on A2B or B2A 
        if mode == "1":
            imageSplice(img2, img1, newPath)
        else:
            imageSplice(img1, img2, newPath)
    print("imageSplice")

    print("finish!")

def formatName():
    DIR = './images'
    k = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name)) and ".png" in name]) + 1
    length = int(len(str(k)))
    for file in os.listdir(DIR):
        name = file.split('.')[0]
        os.rename(os.path.join(DIR, file), os.path.join(DIR, name.zfill(length) + ".png"))
    print("formatName finish!")

if __name__ == "__main__":
    # print(len(sys.argv))
    if len(sys.argv) < 9:
        print('imageProcess.py -s <sketchPath> -r <realScenePath> -m <mode> -d <digit>')
        sys.exit()
    # for arg in sys.argv:
    #     print(arg)
    main(sys.argv[1:])
    # formatName()