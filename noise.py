
import os
import random
import skimage
import cv2
import numpy as np
from PIL import Image, ImageFilter

"""This script can be used to add nose to all the images present in a folder.
   To do so you have to change the paths accordingly to your own local paths"""
folder_dir = "./EnglishImg/English/Img/GoodImg/Bmp/"    #Path to one type of images
os.mkdir('./EnglishImg/English/Img/noisyImag/')
for samples in os.listdir(folder_dir):
    sub_folder_dir=f"{folder_dir}{samples}"
    print(sub_folder_dir)
    os.mkdir(f'./EnglishImg/English/Img/noisyImag/{samples}/')
    for image in os.listdir(sub_folder_dir):
        if (image.endswith(".png")):
            originalImage=cv2.imread(f'{sub_folder_dir}/{image}')
            blurredImage=skimage.util.random_noise(originalImage, mode='gaussian', mean=0.3,seed=None, clip=True)
            blurredImage=skimage.util.random_noise(originalImage, mode='pepper', amount=0.1,seed=None, clip=True)
            noise_img = np.array(255*blurredImage, dtype = 'uint8')
            #cv2.imshow("noisy",noise_img)
            #blurredImage.save(f'./EnglishHnd/English/Hnd/blurImg/{samples}/{image}')
            cv2.imwrite(f'./EnglishImg/English/Img/noisyImag/{samples}/{image}',noise_img)
            #cv2.waitKey(0)
