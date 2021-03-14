import os
import cv2
import numpy as np
from skimage.util import random_noise

listfile = '/Users/albertwu/exp/segmentation-driven-pose/occluded-linemod-testlist.txt'
noise_amount = 0.1
out_dir = f'/Users/albertwu/exp/segmentation-driven-pose/data/OcclusionChallengeICCV2015/RGB-D/rgb_noseg_{noise_amount}/'

try:
    os.mkdir(out_dir)
except FileExistsError:
    pass
with open(listfile, 'r') as file:
    imglines = file.readlines()

for idx in range(len(imglines)):
    imgfile = imglines[idx].rstrip()
    imgfile_name = imgfile.split('/')[-1]
    print(imgfile_name)
    img = cv2.imread(imgfile)
    img = random_noise(img, mode='s&p', amount=noise_amount)  # 0.011
    img = np.array(255 * img, dtype=np.uint8)
    cv2.imwrite(out_dir+imgfile_name, img)
