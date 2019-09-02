# import os, sys
# import random
# import cv2
# import numpy as np

# number = 1000

# def calculateSeamsEnergy(img):
#     seamsEnergy = 0
#     size = img.shape[0]
#     # print(size)
#     # Horizontal
#     for i in range(number):
#         lineNum = random.randint(0, size-1)
#         # print(lineNum)
#         line = img[lineNum:lineNum+1,:]
#         # print(line.shape)
#         # print(int(line[:, 0:1]))
#         for j in range(size-1):
#             seamsEnergy += (float(line[:, j:j+1]) - float(line[:, j+1:j+2])) ** 2
#         # seamsEnergy += np.var(line)
#         # seamsEnergy += np.std(line,ddof=1)
#     # Vertical
#     for i in range(number):
#         lineNum = random.randint(0, size-1)
#         # print(lineNum)
#         line = img[:, lineNum:lineNum+1]
#         # print(line.shape)
#         for j in range(size-1):
#             seamsEnergy += (float(line[j:j+1, :]) - float(line[j+1:j+2, :])) ** 2
#         # seamsEnergy += np.var(line)
#         # seamsEnergy += np.std(line,ddof=1)
    
#     print(seamsEnergy)

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print('need image path')
#         sys.exit()
#     img = cv2.imread(sys.argv[1])
#     img = cv2.resize(img, (3072, 3072))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = img/255
#     calculateSeamsEnergy(img)

import os, sys
import cv2
import numpy as np

def calculateSeamsEnergy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -1)
    # cv2.imwrite("./canny.jpg", cv2.Canny(image, 50, 150))
    # cv2.imwrite("./binary.jpg", binary)
    # binary = cv2.adaptiveThreshold(cv2.Canny(image, 50, 150), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    binary = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(binary,1,np.pi/180,100,minLineLength=80,maxLineGap=10)
    lines1 = lines[:,0,:]
    for x1,y1,x2,y2 in lines1[:]: 
        if (x2-x1) == 0 or (y2-y1)== 0:
            cv2.line(binary,(x1,y1),(x2,y2),(255,255,255),1)

    cv2.imwrite("./binary.jpg", binary)

    rows,cols=binary.shape
    scale = 28
    # print(cols//scale)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,(cols//scale,1))
    eroded = cv2.erode(binary,kernel,iterations = 1)
    # cv2.imwrite("./eroded.jpg", (eroded))
    #cv2.imshow("Eroded Image",eroded)
    dilatedcol = cv2.dilate(eroded,kernel,iterations = 1)
    # cv2.imwrite("./dilatedcol.jpg", (dilatedcol))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,rows//scale))
    eroded = cv2.erode(binary,kernel,iterations = 1)
    dilatedrow = cv2.dilate(eroded,kernel,iterations = 1)
    # cv2.imwrite("./dilatedrow.jpg", dilatedrow)


    merge = cv2.add(dilatedcol,dilatedrow)
    # print(merge)
    cv2.imwrite("./merge.jpg", merge)
    print(np.sum(merge))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('need image path')
        sys.exit()
    
    img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, (3072, 3072))
    calculateSeamsEnergy(img)