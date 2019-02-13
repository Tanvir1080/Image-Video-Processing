from PIL import Image
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

# Tanvir Islam 
# nid ta840968
# ucfid 3779383
# Spring 2019

"""The 3x3 filter removes noise from our given image and smoothes it out 
which results in a blurry looking image. 
The 5x5 filter does the same thing as the 3x3 except it smoothes it out 
more and results in more blurrying."""

def BoxFilter (size, img): 

    # our kernel is made of 1's and dimension is the size we give it.
    # it is then normalized by dividing by the square of the size 
    kernel = np.ones((size,size), np.float32)/(np.power(size, 2))
    dst = ndimage.convolve(img, kernel)
    return dst

def BoxFilterParent(size, img):
    removedNoise = BoxFilter(size, img)
    return Image.fromarray(removedNoise)
    
    
def main():
    cvimg1 = Image.open("./Images/image1.png")
    cvimg2 = Image.open("./Images/image2.png")
    
    box1With3 = BoxFilterParent(3, cvimg1)
    box1With5 = BoxFilterParent(5, cvimg1)
    box2With3 = BoxFilterParent(3, cvimg2)
    box2With5 = BoxFilterParent(5, cvimg2)

    plt.subplot(221),plt.imshow(box1With3),plt.title('Image 1 with 3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(box1With5),plt.title('Image 1 with 5')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(box2With3),plt.title('Image 2 with 3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(box2With5),plt.title('Image 2 with 5')
    plt.xticks([]), plt.yticks([])
    plt.show()

main()