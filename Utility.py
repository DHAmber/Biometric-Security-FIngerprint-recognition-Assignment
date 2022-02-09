import cv2

blockSize=16
DesiredMean=float(100)
DesiredVarience=float(100)
def ShowImage(image,msg):
    cv2.imshow(msg, image)
    cv2.waitKeyEx()