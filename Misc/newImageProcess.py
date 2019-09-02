import sys, getopt
import os
import cv2
import numpy as np

row = 7
column = 5
size = 3072

def imageResize(oldPath, width, height):
    # print("111")
    # print(oldPath)
    img = cv2.imread(oldPath)
    # print(img.shape)
    img2 = cv2.resize(img,(width, height),)
    return img2

def imageCut(img, size, folder, k, digit):
    length = row * 12 * size
    # print(img.shape)
    num1 = int (length / size)
    num2 = int ((column * 12 * size) / size)
    for i in range(0, num1):
        for j in range(0, num2):
            newImg = img[i * size : (i + 1)* size, j * size: (j + 1) * size, :]
            cv2.imwrite(folder + str(k).zfill(digit) + ".png", newImg)
            k += 1

def imageSplice(img1, img2, newPath):
    newImage = np.hstack([img1, img2])
    cv2.imwrite(newPath, newImage)

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
    
    sketchList = [name for name in os.listdir(sketchPath) if os.path.isfile(os.path.join(sketchPath, name)) and (".png" in name or "tif" in name)]
    sketchList.sort()
    realSceneList = [name for name in os.listdir(realScenePath) if os.path.isfile(os.path.join(realScenePath, name)) and ".jpg" in name]
    realSceneList.sort()
    if len(sketchList) != len(realSceneList):
        print("images number does not match")
        sys.exit()
    
    totalNumber = len(sketchList)
    # print(totalNumber)
    # new_img = np.full((size * row, size * column, 3), 255.0)
    # cv2.imwrite("./tttt.png", new_img)
    print("imageCombine start!")

    new_sketch = cv2.imread("./base.png")
    new_realScene = cv2.imread("./base.png")

    if not os.path.isdir("./sketch"):
        os.mkdir("./sketch")
    if not os.path.isdir("./realScene"):
        os.mkdir("./realScene")
    if not os.path.isdir("./images"):
        os.mkdir("./images")
    
    # print(sketchList)
    # print(realSceneList)
    k = 0
    for j in range(0, column):
        for i in range(0, row):
            new_sketch[(row-i-1)*size: (row-i)*size, j*size:(j+1)*size, :] = imageResize(sketchPath + sketchList[k], size, size)[:, :, :]
            new_realScene[(row-i-1)*size: (row-i)*size, j*size:(j+1)*size, :] = imageResize(realScenePath + realSceneList[k], size, size)[:, :, :]
            k += 1

    # cv2.imwrite("./sketch.png", new_sketch)
    # cv2.imwrite("./realScene.png", new_realScene)
    print("imageCombine finish!")

    print("imageCut start!")
    imageCut(new_sketch, 256, "./sketch/", 0, digit)
    imageCut(new_realScene, 256, "./realScene/", 0, digit)
    print("imageCut finish!")

    print("imageSplice start!")
    for i in range(0, row * column * 144):
        newPath = "./images/" + str(i).zfill(digit) + ".png"
        img1 = cv2.imread("./sketch/" + str(i).zfill(digit) + ".png")
        img2 = cv2.imread("./realScene/" + str(i).zfill(digit) + ".png")
        # change img1 and img2 based on A2B or B2A 
        if mode == "1":
            imageSplice(img2, img1, newPath)
        else:
            imageSplice(img1, img2, newPath)
    print("imageSplice finish!")

    print("finish!")

if __name__ == "__main__":
    # print(len(sys.argv))
    if len(sys.argv) < 9:
        print('imageProcess.py -s <sketchPath> -r <realScenePath> -m <mode> -d <digit>')
        sys.exit()
    # for arg in sys.argv:
    #     print(arg)
    main(sys.argv[1:])
    # formatName()