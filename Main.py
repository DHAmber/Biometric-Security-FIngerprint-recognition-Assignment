import numpy as np
import cv2 as cv              #cv2 is opencv( to process the image,image processing related functions)
import glob as fileOpener         #glob reads a pattern,if we wud have want all fingerprint files,then it is used
import Processing as process
import Orientation as o
import Utility as utl



if __name__ == '__main__':
    img_path=fileOpener.glob('FingerPrintDB/101_4.tif')
    img=np.array(cv.imread(img_path[0],0))              #first 0 is image position,second 0 converts to grayscale,#converts image to numpy array, soince image is 2D so converts it to 2D array
    cv.imshow('Original',img )
    cv.waitKey(0)
    cv.destroyAllWindows()
    normalized_img=process.NormalizedImage(img)
    cv.imshow('Normalized', normalized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    segemented_Img,mask=process.GetSegmentedImage(normalized_img)
    cv.imshow('Segmented', segemented_Img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    Orientation=o.GetOrientation(normalized_img,utl.DesiredMean,utl.DesiredVarience)
    process.gabor(segemented_Img,Orientation)




