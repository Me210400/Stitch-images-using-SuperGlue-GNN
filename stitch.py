import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# generating the necessary txt file to input for the super glue algorithm
img_name = 'frame' # set of outdoor images
num_images = 360
# Order of the images. To stitch left and right images as depicted in the below
order = range(num_images -1,0,-1) 
# with open('orig.txt', 'w') as file:
#     for i in order:
#         file.write("{img}{:02}.jpg {img}{:02}.jpg\n".format(i,i-1, img = img_name))

#generatig the npz files for extract matching information
npz_files = ["{img}{:02}_{img}{:02}_matches.npz".format(i,i-1, img = img_name) for i in order]
for file in npz_files:
    path = 'orig_output/'+file
    npz = np.load(path)
#print(npz.files)

# extracting information from the npz files 
def loadNPZ(npz_file):    
    npz = np.load('orig_output/'+ npz_file)
    point_set1 = npz['keypoints0'][npz['matches']>-1]
    matching_indexes =  npz['matches'][npz['matches']>-1] # -1 if the keypoint is unmatched
    point_set2 = npz['keypoints1'][matching_indexes]
    #print("Number of matching points for the findHomography algorithm:")
    #print("In left  image:", len(point_set1),"\nIn right image:", len(point_set2))
    return point_set1, point_set2

def pltSourceImages(imageSet):    
    im_left = cv.imread('frames/frame{:02}.jpg'.format(imageSet),cv.IMREAD_ANYCOLOR)
    im_right = cv.imread('frames/frame{:02}.jpg'.format(imageSet -1),cv.IMREAD_ANYCOLOR)
    
    # Marking the detected features on the two images.
    for point in point_set1.astype(np.int32):
        cv.circle(im_left, tuple(point), radius=8, color=(255, 255, 0), thickness=-1)

    for point in point_set2.astype(np.int32):
        cv.circle(im_right, tuple(point), radius=8, color=(255, 255, 0), thickness=-1)

    fig = plt.figure(figsize = (10,10))
    # plt.subplot(121),plt.imshow(im_left, cmap='gray', vmin = 0, vmax = 255)
    # plt.subplot(122),plt.imshow(im_right, cmap='gray', vmin = 0, vmax = 255)
    # plt.show()





def plotMatches(imageSet):
    plt.figure(figsize=(10,10))
    matched_points = cv.imread('orig_output/frame{:02}_frame{:02}_matches.png'.format(imageSet, imageSet -1),cv.IMREAD_ANYCOLOR)
    # plt.imshow(matched_points, cmap='gray', vmin = 0, vmax = 255)
    # plt.show()

#homography = []
for imgSet in range(num_images-1,1,-1):  
    # loading points
    point_set1, point_set2 = loadNPZ(npz_files[num_images-1 -imgSet])
    pltSourceImages(imgSet)
    plotMatches(imgSet)    
    # getting the required source images
    im_left = cv.imread('frames/frame{:02}.jpg'.format(imgSet),cv.IMREAD_ANYCOLOR)
    im_right = cv.imread('frames/frame{:02}.jpg'.format(imgSet -1),cv.IMREAD_ANYCOLOR)
    #find Homography between two source images
    H, status = cv.findHomography(point_set1, point_set2, cv.RANSAC, 5.0)
    #homography.append(H)
    # with open ('homography.txt', 'w') as f:
    #     f.write(str(H))
    #     f.write("\n")
    #     f.write('--------------------------------------------------------------')
    # Prints the Homography matrix that transform left image to right image
    #print(H) 
    #print(homography)
    # Applies a homogeneous transformation to an image.
    # To transform the right image to left we need to consider the inverse.
    panorama = cv.warpPerspective(im_right, np.linalg.inv(H), (im_right.shape[1] + im_left.shape[1], im_right.shape[0])) 
    print(panorama.shape)
   

 
    #print(im_right)
    # plt.figure(figsize=(10,10))
    # plt.imshow(panorama, cmap='gray', vmin = 0, vmax = 255)
    # plt.show()
    #print(im_left.shape)
    panorama[0:im_left.shape[0], 0:im_left.shape[1]] = im_left
    #print(panorama[0:im_left.shape[0], 0:im_left.shape[1]])
    # plt.figure(figsize=(10,10))
    # plt.imshow(im_left, cmap = 'gray', vmin = 0, vmax = 255)
    #plt.show()
    # plt.imshow(panorama, cmap='gray', vmin = 0, vmax = 255)
    cv.imwrite('porc/res%d.jpg' % imgSet, panorama)
    #plt.show()    
    print("-"*100)
    # hmg_matrix = np.array(homography)
    # with open ("homo.txt", "a") as f:
    #     content = str(hmg_matrix)
    #     f.write(content + '\n')
      
#     hmg_matrix = np.array(homography)
#     reshaped = hmg_matrix.reshape(hmg_matrix.shape[0], -1)
# np.savetxt("hmg.txt", reshaped)





     
