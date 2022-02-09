import numpy as np
import cv2 as cv
import math
import Utility as U
from datetime import datetime

def product(x,y):
    return 2*x*y

def diff(x,y):
    return x**2-y**2

def CalculateAngel(i,j,Gx,Gy,x,y):
    w = U.blockSize
    a=0
    b=0
    for l in range( i,min(i+w,x-1)):
        for m in range(j,min(j+w,y-1)):
            _Gx=Gx[l][m]
            _Gy=Gy[l][m]
            xy=product(_Gx,_Gy)
            x_y=diff(_Gx,_Gy)
            a=+xy
            b+=x_y
    return 1/2* math.atan2(a,b)

def GetOrientation(img,DesiredMean,DesiredVarience):
    result=np.zeros_like(img)
    sobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    xSobel = np.array(sobel).astype(np.int)
    ySobel = np.transpose(xSobel).astype(np.int)

    x,y=np.shape(img)
    Gx=cv.filter2D(img,-1,xSobel)
    Gy = cv.filter2D(img, -1, ySobel)

    for i in range(1,x,U.blockSize):
        for j in range(1,y,U.blockSize):
            result[i][j]=CalculateAngel(i,j,Gx,Gy,x,y)
    return result


