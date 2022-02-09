import numpy as np
import cv2 as cv
import math
import Utility as U


def NormalizedImage(img):
    M=np.mean(img)
    V=np.std(img)**2
    result=np.copy(img)
    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            a=math.sqrt((float(100) * ((img[i,j] - M) ** 2)) / V)
            if(img[i,j]<M):
                b= float(100)+a
            else:
                b= float(100)-a
            result[i,j]=b

    return result

def GetSegmentedImage(im):
    threshold = 0.2
    blocksize = 16
    segmented_image = im.copy()
    image_variance = np.zeros(im.shape)
    threshold = np.std(im)* threshold
    (length,width)=np.shape(im)
    mask=np.ones_like(im)
    for i in range(0,width,blocksize):
        for j in range(0,length,blocksize):
            if (i+blocksize<width):
                x=i+blocksize
            else:
                x=width
            if (j+blocksize<length):
                y=j+blocksize
            else:
                y=length
            box = [i, j, x, y]
            image_variance[box[1]:box[3], box[0]:box[2]]=np.std(im[box[1]:box[3], box[0]:box[2]])

    mask[image_variance < threshold] = 0
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (blocksize*2 , blocksize*2 ))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k)

    x, y = im.shape
    for i in range(0,x):
        for j in range(0,y):
            segmented_image[i,j]=segmented_image[i,j]*mask[i,j]
    return segmented_image,mask

def xy(x,y):
    return 2*x*y
def x_y(x,y):
    return x**2-y**2
def CaclculateAngel(i,j,ArGx,ArGy,x,y):
    XY2 = 0
    X_Y = 0

    for b in range(j, j + U.blockSize if j + U.blockSize < y - 1 else y - 1):
        for a in range(i, i + U.blockSize if i + U.blockSize < x - 1 else x - 1):
            Gx = ArGx[a, b]
            Gy = ArGy[a, b]
            XY2 += xy(Gx, Gy)
            X_Y += x_y(Gx, Gy)

    return (math.pi + math.atan2(round(XY2), round(X_Y))) / 2


def gabor(img,angels):
    res=np.zeros_like(img)
    x,y=np.shape(img)
    w=U.blockSize
    for row in range(1,x,U.blockSize):
        for col in range(1,y,U.blockSize):
            g_kernel = cv.getGaborKernel((row,col), 8.0, angels[row][col], 10.0, 0.5, 0, ktype=cv.CV_32F)
            image_block = img[row:row + w][:, col:col + w]
            filtered_img = cv.filter2D(image_block, cv.CV_8UC3, g_kernel)
            res[row:row + w][:, col:col + w]=filtered_img
    cv.imshow('Gabor filterd image', res)
    cv.waitKey(0)
    cv.destroyAllWindows()