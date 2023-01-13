
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


IMG_NAME = 'frame' # set of outdoor images
NUM_IMAGES = 360
# Order of the images. To stitch left and right images as depicted in the below
order = range(0,NUM_IMAGES -1) 

#generatig the npz files for extract matching information
npz_files = ["{img}{:02}_{img}{:02}_matches.npz".format(i+1,i, img = IMG_NAME) for i in order]
for file in npz_files:
    path = 'frames/output/'+file
    npz = np.load(path)
#print(npz_files[0])

# extracting information from the npz files 
def loadNPZ(npz_file):    
    npz = np.load('frames/output/'+ npz_file)
    point_set1 = npz['keypoints0'][npz['matches']>-1]
    matching_indexes =  npz['matches'][npz['matches']>-1] # -1 if the keypoint is unmatched
    point_set2 = npz['keypoints1'][matching_indexes]
    keypoints_im_left = np.float32(point_set1)
    keypoints_im_right = np.float32(point_set2)
    # print("Number of matching points for the findHomography algorithm:")
    # print("In left  image:", len(point_set1),"\nIn right image:", len(point_set2))
    return keypoints_im_left, keypoints_im_right

def pltSourceImages(imageSet):    
    im_left = cv.imread('frames/frame{:02}.png'.format(imageSet + 1),cv.IMREAD_ANYCOLOR)
    im_right = cv.imread('frames/frame{:02}.png'.format(imageSet),cv.IMREAD_ANYCOLOR)
    
    # Marking the detected features on the two images.
    for point in keypoints_im_left.astype(np.int32):
        cv.circle(im_left, tuple(point), radius=8, color=(255, 255, 0), thickness=-1)

    for point in keypoints_im_right.astype(np.int32):
        cv.circle(im_right, tuple(point), radius=8, color=(255, 255, 0), thickness=-1)

    #fig = plt.figure(figsize = (10,10))
    # plt.subplot(121),plt.imshow(im_left, cmap='gray', vmin = 0, vmax = 255)
    # plt.subplot(122),plt.imshow(im_right, cmap='gray', vmin = 0, vmax = 255)
    # plt.show()


def plotMatches(imageSet, keypoints_im_left, keypoints_im_right):
    #plt.figure(figsize=(10,10))
    matched_points = cv.imread('frames/output/frame{:02}_frame{:02}_matches.jpg'.\
                     format(imageSet + 1, imageSet),cv.IMREAD_ANYCOLOR)
    # print(matched_points)
    # input()
    # print(matched_points)
    # input()
    # plt.imshow(matched_points, cmap='gray', vmin = 0, vmax = 255)
    # plt.show()
    # if len(matched_points) > 4:
    #     points_im_left = np.float32(matched_points)
    #     points_im_right = np.float32(matched_points)
    # return points_im_left, points_im_right


#homography = []
for imgSet in range(0, NUM_IMAGES - 1):  
    #print(imgSet)
    # loading points
    keypoints_im_left, keypoints_im_right  = loadNPZ(npz_files[imgSet])
    #print(point_set1, point_set2)
    pltSourceImages(imgSet)
    plotMatches(imgSet, keypoints_im_left, keypoints_im_right)    
    # getting the required source images
    im_left = cv.imread('frames/frame{:02}.png'.format(imgSet + 1), cv.COLOR_BGR2RGB)

    im_right = cv.imread('frames/frame{:02}.png'.format(imgSet ),cv.COLOR_BGR2RGB)
    # print(im_right)

    #find Homography between two source images
    (H, status) = cv.findHomography(keypoints_im_left, keypoints_im_right, cv.RANSAC, 4.0) 
    # inverse = np.linalg.inv(H)
    #homography.append(H)
    # Prints the Homography matrix that transform left image to right image
    # print(H) 
    # print(type(H))
    # input()
    # width = im_left.shape[1] + im_right.shape[1]
    # height = max(im_left.shape[0], im_right.shape[0])
    # Applies a homogeneous transformation to an image.
    # To transform the right image to left we need to consider the inverse.
    #print(H)
    panorama = cv.warpPerspective(im_left, H, (2800,1600 )) 
    # print(panorama)
    # input()
 
    # plt.figure(figsize=(10,10))
    # plt.imshow(panorama, cmap='gray', vmin = 0, vmax = 255)
    panorama[0:im_right.shape[0], 0:im_right.shape[1]] = im_right
    # print(panorama)
    # input()
    cv.imwrite('frames/res_2/%d.png' % imgSet, panorama)
    # plt.figure(figsize=(20,10))  
    # plt.imshow(panorama, cmap='gray', vmin = 0, vmax = 255)
    # plt.show()    
    print("-"*100)
    #hmg_matrix = np.array(homography)
    # print(hmg_matrix.shape)
    # input()
    # with open ("frames/homo.txt", "a") as f:
    #     content = str(hmg_matrix)
    #     f.write(content + '\n')
#     hmg_matrix = np.array(homography)
#     reshaped = hmg_matrix.reshape(hmg_matrix.shape[0], -1)
# np.savetxt("frames/homography.txt", reshaped)
# print(hmg_matrix.shape)
# input()






