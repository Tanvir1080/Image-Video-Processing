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


Comparison with Question 1 and 2 
We can see that for image 1, both the median filter and the gaussian filter to a better job of removing 
the salt and pepper noise at the lower kernels, wuch as 3 or 5. 
As we go higher up in kernel/sigma size, the gaussian filter starts to blur the images a lot worse than the median does.
This is because the optimal sigma for Gaussian is between 0.8 and 1 anyway. 
For both image 1 and image 2 it can be seen that Median filter starts to round corners and make edges 
more jagged that either box or gaussian filter, making it less optimal for edge detection.
For both Gaussian and Median image 1 and 2 can be hardly distinguished even after smoothing, which is not the case 
for Box. 
I would say that if sigma is kept closer toward its optimal condition then the Gaussian Filter would be the best one
for both image1 and imag2.  
This is because the smoothing is much better than the Box filter and it doesn't make the edges more jagged and rough like 
the Median Filter does. """

def GaussianFilter(img, sigma): 

    # dynamically modify the size of our kernel so that the images do not become dark 
    size = sigma * 6

    if(size % 2 == 0):
        size = size + 1
    
    # the offset helps us calculate our values to put inside the kernel 
    offset = (size // 2) * -1
    
    kernel = np.ndarray((size,size))
    coeff = 1 / (2 * np.square(sigma) * np.pi)
    
    for i in range(size):
        for j in range(size):

            x = i + offset
            y = j + offset

            # apply the gaussian formula to generate our values in each index of the kernel
            # for convolution
            power = -1 * ((np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            ePower = np.exp(power)
            kernel[i][j] = ePower * coeff
    
    dst = ndimage.convolve(img, kernel)

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