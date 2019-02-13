from PIL import Image 
import numpy as np
from scipy import ndimage
import math
from matplotlib import pyplot as plt

# Tanvir Islam 
# nid ta840968
# ucfid 3779383
# Spring 2019

"""Sobel Filter is an edge detection algorithm at its core even if it's called a filter
The way its implemented is using 2 separate kernels and then combining them to get all the edges at the end

Discussion 
This edge detection is done without any smoothing on the imaages since the instructions did not say 
to do so in the PDF.  Therefore, they will be discussed as so. 
Due to the presence of tremendous salt and pepper noise in image1 it is hard to see sobel's 
work but it is visible. 
It can be seen really well with image2 which shows all the edges as white lines in the image. 
In fact, you are able to see edges with this algorithm that I didn't even see when I looked at the raw images myself the first time. 
An interesting to note is that even though there is lots of noise in image1, sobel still works to show the size of the noise, 
which is something hard to see in the original image. """

def SobelFilter(img): 

    # We make our kernels to go in the x and y direction separately and then convolve 
    # them both on the image
    kernelX = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    kernelY = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    
    dstX = ndimage.convolve(img, kernelX)
    dstY = ndimage.convolve(img, kernelY)
    dst = np.ndarray((256,256))
    
    # this is how we combine our two kernel convolutions
    # by taking the magnitude and storing that in a new array 
    for i in range(256):
        for j in range(256):
            dst[i][j] = math.sqrt((math.pow(dstX[i][j], 2) + math.pow(dstY[i][j], 2)))

    return dst

def SobelFilterParent(img):
    arr = np.array(img, dtype=float)
    removedNoise = SobelFilter(arr)
    return Image.fromarray(removedNoise)
    

def main():
    img1 = Image.open("./Images/image1.png")
    img2 = Image.open("./Images/image2.png")

    sob1 = SobelFilterParent(img1)
    sob2 = SobelFilterParent(img2)

    plt.subplot(121),plt.imshow(sob1),plt.title('Image 1')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(sob2),plt.title('Image 2')
    plt.xticks([]), plt.yticks([])
    plt.show()

main()