import glob
import numpy as np

files = glob.glob(r'D:\Imagery\HLSv2.0\*\**\*.tif', recursive=True)

np.savetxt(r'D:\Imagery\HLSv2.0\HLS_v2.0_inventory.txt', files, fmt='%s')