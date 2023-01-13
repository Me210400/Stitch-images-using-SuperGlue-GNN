import os
from natsort import natsorted

folder_path = "frames/res_2"

# Get all the files in the folder
files = os.listdir(folder_path)

# Sort the files using natsort
files = natsorted(files)
print(files)
