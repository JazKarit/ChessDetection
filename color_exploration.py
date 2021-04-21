# Jaskrit Singh
# CSCI 4831
# Project
# Ioana Fleming 

import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import imutils

# Explore the color channels in an image. 
# Mainly focused on choosing between RGB and HSV.

def crop_resize_img(img):
    """ Crop and resize image to just the chessboard based on my camera setup. """
    img = img[int(img.shape[0]/70):img.shape[0]-int(img.shape[0]/40),int(img.shape[1]/5):img.shape[1]-int(img.shape[1]/5)]
    scale_percent = 20 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img,dim)
    return img

img = cv2.imread('game1\ezgif-frame-001.jpg')
# img = cv2.imread('game2\\frame23.jpg')
img = crop_resize_img(img)


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img)


hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)


plt.subplot(2, 3, 1)
plt.title("Red")
plt.imshow(r, cmap="gray")
plt.subplot(2, 3, 2)
plt.title("Green")
plt.imshow(g, cmap="gray")
plt.subplot(2, 3, 3)
plt.title("Blue")
plt.imshow(b, cmap="gray")
plt.subplot(2, 3, 4)
plt.title("Hue")
plt.imshow(h, cmap="gray")
plt.subplot(2, 3, 5)
plt.title("Saturation")
plt.imshow(s, cmap="gray")
plt.subplot(2, 3, 6)
plt.title("Value")
plt.imshow(v, cmap="gray")
plt.show()

# Sources:
# https://realpython.com/python-opencv-color-spaces/