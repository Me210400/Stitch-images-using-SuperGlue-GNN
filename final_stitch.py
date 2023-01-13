
import os
import glob
import cv2 
import natsort
import numpy as np


def ReadImage(folder):
    image_list = os.listdir(folder)
    image_list = natsort.natsorted(image_list)
    sorted_images = []
    for image in image_list:
        img = cv2.imread(os.path.join(folder, image))
        sorted_images.append(img)
    return sorted_images



def StitchImages(im_left, im_right, homography):
     panorama = cv2.warpPerspective(im_right, homography, (2800,1600 )) 
     panorama[0:im_left.shape[0], 0:im_left.shape[1]] = im_left
     return panorama

if __name__ == "__main__":
    # Reading images.
    Images = ReadImage("frames/res_2")
    #print(len(Images))
    homo_load = np.loadtxt("frames/homography.txt")
    # print(len(homo_load))
    # input()
    homo = homo_load.reshape(homo_load.shape[0], homo_load.shape[1] // 3, 3)
    
    BaseImage = Images[0]
    for i in range(1, len(Images)):
        StitchedImage = StitchImages(BaseImage, Images[i], homo[i-1])

        BaseImage = StitchedImage.copy()    

cv2.imwrite("frames/final/13.01.2023.png", BaseImage)