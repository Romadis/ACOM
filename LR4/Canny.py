import cv2
import numpy as np
import math

def Image(path, kernelSize = 5, sigmaX=10, sigmaY=10, sizeX=640, sizeY=640):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (sizeX, sizeY))
    cv2.imshow("Gray img", img)

    imgGaussian = cv2.GaussianBlur(img, (kernelSize, kernelSize), sigmaX=sigmaX, sigmaY=sigmaY)
    print(imgGaussian)
    cv2.imshow("GaussBlur img", imgGaussian)

    grads = Gradients(imgGaussian)
    print(grads)
    lengths = GradLengths(imgGaussian, grads)
    print(lengths)
    corners = Corners(imgGaussian, grads)
    print(corners)

    suppressed_img = supressNotMax(lengths, corners)
    cv2.imshow('Suppressed Image', suppressed_img)

    edgeImg = checkThreshAndEdge(imgGaussian, suppressed_img, lengths, 10)
    cv2.imshow('Contours Image', edgeImg)
    print(edgeImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Gradients(img):
    gradientMatrix = []
    for x in range(1, img.shape[0]-1):
        matrixRow = []
        for y in range(1, img.shape[1] - 1):
            Gx = -int(img[x - 1][y - 1]) - 2*int(img[x][y-1]) - int(img[x + 1][y - 1]) + \
                 int(img[x - 1][y + 1]) + 2*int(img[x][y + 1]) + int(img[x + 1][y + 1])
            Gy = -int(img[x - 1][y - 1]) - 2*int(img[x - 1][y]) - int(img[x - 1][y + 1]) + \
                 int(img[x + 1][y - 1]) + 2*int(img[x + 1][y]) + int(img[x + 1][y + 1])
            matrixRow.append((Gx, Gy))
        gradientMatrix.append(matrixRow)
    return gradientMatrix
def GradLengths(img, grads):
    res = np.zeros((img.shape[0], img.shape[1]))
    k = 0
    for i in range(1, img.shape[0]-1):
        l = 0
        for j in range(1, img.shape[1]-1):
            res[i][j] = math.sqrt(grads[k][l][0]**2+grads[k][l][1]**2)
            l = l + 1
        k = k + 1
    return res

def Corner(grad):
    tang = grad[1]/grad[0] if grad[0] != 0 else 999
    if grad[0] > 0 and grad[1] < 0 and tang < -2.414 or grad[0] < 0 and grad[1] < 0 and tang > 2.414:
        return 0
    elif grad[0] > 0 and grad[1] < 0 and tang < -0.414:
        return 1
    elif grad[0] > 0 and grad[1] < 0 and tang > -0.414 or grad[0] > 0 and grad[1] > 0 and tang < 0.414:
        return 2
    elif grad[0] > 0 and grad[1] > 0 and tang < 2.414:
        return 3
    elif grad[0] > 0 and grad[1] > 0 and tang > 2.414 or grad[0] < 0 and grad[1] > 0 and tang < -2.414:
        return 4
    elif grad[0] < 0 and grad[1] > 0 and tang < -0.414:
        return 5
    elif grad[0] < 0 and grad[1] > 0 and tang > -0.414 or grad[0] < 0 and grad[1] < 0 and tang < 0.414:
        return 6
    elif grad[0] < 0 and grad[1] < 0 and tang < 2.414:
        return 7
    if (grad[0] == 0):
        if (grad[1] > 0):
            return 4
        elif (grad[1] <= 0):
            return 0
    else:
        if (grad[1] > 0):
            return 2
        elif (grad[1] <= 0):
            return 6

def Corners(img, grads):
    corners = np.zeros((img.shape[0], img.shape[1]))
    k = 1
    for i in range(len(grads)):
        l = 1
        for j in range(len(grads[0])):
            corners[k][l] = Corner(grads[i][j])
            l = l + 1
        k = k + 1
    return corners

def supressNotMax(gradsLenths, corners):
    height, width = gradsLenths.shape
    suppressed = np.zeros_like(gradsLenths)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = corners[x][y]

            if angle == 0 or angle == 4:
                q = gradsLenths[x+1][y]
                r = gradsLenths[x-1][y]
            elif angle == 1 or angle == 5:
                q = gradsLenths[x-1][y+1]
                r = gradsLenths[x + 1][y - 1]
            elif angle == 2 or angle == 6:
                q = gradsLenths[x][y+1]
                r = gradsLenths[x][y-1]
            elif angle == 3 or angle == 7:
                q = gradsLenths[x+1][y+1]
                r = gradsLenths[x-1][y-1]

            if gradsLenths[x][y] >= q and gradsLenths[x][y] >= r:
                suppressed[x][y] = 255
            else:
                suppressed[x][y] = 0

    return suppressed
def checkThreshAndEdge(img, filteredImg, gradientsLength, boundPath1=10, boundPath2 = 25):
    maxGradient = np.max(gradientsLength)
    print(maxGradient)
    lower_bound = maxGradient / boundPath1
    upper_bound = maxGradient / boundPath2
    img_border_filter = np.zeros(img.shape)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            gradient = gradientsLength[i][j]
            if (filteredImg[i][j] == 255):
                if (gradient >= lower_bound and gradient <= upper_bound):
                    print(1)
                    flag = False
                    # проверим соседние пиксели текусщего пикселя
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            if (flag):
                                break
                            if (filteredImg[i + k][j + l] == 255 and filteredImg[i + k][j + l] >= lower_bound):
                                flag = True
                                break
                    if (flag):
                        filteredImg[i][j] = 255
                elif (gradient > upper_bound):
                    img_border_filter[i][j] = 255
    return img_border_filter

Image("1.jpg")