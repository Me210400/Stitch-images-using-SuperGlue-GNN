

# generating the necessary txt file to input for the super glue algorithm
img_name = 'frame' # set of outdoor images
num_images = 360
# Order of the images. To stitch left and right images as depicted in the below
order = range(num_images -1,0,-2) 
with open('my.txt', 'w') as file:
    for i in order:
        file.write("{img}{:02}.jpg {img}{:02}.jpg\n".format(i,i-1, img = img_name))

