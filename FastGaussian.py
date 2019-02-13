from __future__ import division
import numpy as np 
from scipy import ndimage
from PIL import Image
from matplotlib import pyplot as plt

# Tanvir Islam 
# nid ta840968
# ucfid 3779383
# Spring 2019

"""The gaussian filter applies what is essentially a weighted average to our images
For each size x size section in the image, the central pixel of our kernel has the highest weight.
and weight is less significant as you go out and away from the image.
Applying a greater sigma (variance) on reach results in dramatically increased smoothing and blurring of the image.
This is comparable to our own eyes because a larger sigma is like looking at a broad scene 
without focusing on a particular point, which is the opposite of using a smaller sigma.

This question differs from question 3 in that the kernels used are different 
Here we are using 2 singular dimension kernels and a single for loop instead of a nested one. 
This makes our runtime much much faster than regular gaussian because we are O(2n) instead of O(n^2)
As for the smoothing itself for the image, there is no difference between this and regular Gaussian because
they both still do the same thing, just in a different way. """

def GaussianFilter(img, sigma): 

    size = sigma * 6

    if(size % 2 == 0):
        size = size + 1
    
    offset = (size // 2) * -1
    
    # we make our two singular dimension kernels 
    kernelY = np.ones((size, 1))
    kernelX = np.ones((1,size))

    # also note here that the formula is slightly different from regular gaussian 
    coeff = 1 / ((np.sqrt(2 * np.pi)) * sigma)
    stDev = 2 * np.square(sigma)

    # we work with a single array, just referencing two arrays at the same time, both O(n)
    for i in range(size):
        kernelX[0][i] = coeff * np.exp(-(np.square(offset + i)) / stDev)
        kernelY[i][0] = coeff * np.exp(-(np.square(offset + i)) / stDev)
    
    """in order to avoid the "scratches" present in each image convolution
    we convolve in the X direction and then we use that same image and 
    convolve over it using our Y kernel to get our final image. """
    dst = ndimage.convolve(img, kernelX)
    dst = ndimage.convolve(dst, kernelY)

    return dst

def GaussianFilterParent(img, sigma, name):
    arr = np.array(img, dtype=float)
    removedNoise = GaussianFilter(arr, sigma)
    return Image.fromarray(removedNoise)
    

def main():
    img1 = Image.open("./Images/image1.png")
    img2 = Image.open("./Images/image2.png")

    gauss1With3 = GaussianFilterParent(img1, 3, "Img1With3") 
    gauss1With5 = GaussianFilterParent(img1, 5, "Img1With5")
    gauss1With10 = GaussianFilterParent(img1, 10, "Img1With10")
 
    gauss2With3 = GaussianFilterParent(img2, 3, "Img2With3")
    gauss2With5 = GaussianFilterParent(img2, 5, "Img2With5")
    gauss2With10 = GaussianFilterParent(img2, 10, "Img2With10")

    plt.subplot(231),plt.imshow(gauss1With3),plt.title('Image 1 with 3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(gauss1With5),plt.title('Image 1 with 5')
    plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(gauss1With10),plt.title('Image 1 with 10')
    plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(gauss2With3),plt.title('Image 2 with 3')
    plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(gauss2With5),plt.title('Image 2 with 5')
    plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(gauss2With10),plt.title('Image 2 with 10')
    plt.xticks([]), plt.yticks([])
    plt.show()

main()