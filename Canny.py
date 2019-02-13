from __future__ import division
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import ndimage

# Tanvir Islam 
# nid ta840968
# ucfid 3779383
# Spring 2019

""" It's always important to smooth the images and out and convert to grayscale
    when doing edge detection.  Gaussian Filter was used as the smoothing Filter 
    since it seems to be working the best from the other functions done in this 
    assignment.  Sigma = 1 was chosen as the optimal sigma because it results in 
    the least amount of blurring.  If you increase sigma, the images get blurrier
    and this makes the edges more round and thick so it's harder to detect them 
    as the Canny algorithm progresses.
    When running this algorithm on image1 and image2 different sigmas and thresholds were tested. 
    The sigma of 1 is the best still due to the nature of gaussian filter and that optimal sigma 
    are between 0.8 and 1 regardless. 
    The threshold of about 30 was also found to be optimal however due to time constraints, there
    were a few pixels that seem to be missing from the corners of image1 and image2 that 
    were not able to be adjusted for. """

""" This is our fast gaussian function from FastGaussian.py which implements 
    1 dimensional kernel to do 2D filtering. """
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

    for i in range(size):
        kernelX[0][i] = coeff * np.exp(-(np.square(offset + i)) / stDev)
        kernelY[i][0] = coeff * np.exp(-(np.square(offset + i)) / stDev)
    
    """in order to avoid the "scratches" present in each image convolution
    we convolve in the X direction and then we use that same image and 
    convolve over it using our Y kernel to get our final image."""

    dst = cv2.filter2D(img, -1, kernelX)
    dst = cv2.filter2D(dst, -1, kernelY)

    return dst

def GaussianFilterParent(img, sigma):
    arr = np.array(img, dtype=float)
    removedNoise = GaussianFilter(arr, sigma)
    return removedNoise

""" We obtain our gradient images in both directions as well as the final gradient.
    we also obtain directions at each pixel for our gradient to determine the 
    orientation for use with the next step of canny. """
def CentralGradient(img, width, height):
    # img = cv2.imread(name)
    kernelX = np.array([[1,0,-1], [2, 0, -2], [1, 0, -1]]) 
    kernelY = np.array([[1,2,1], [0,0,0], [-1, -2, -1]]) 

    magX = cv2.filter2D(img, -1, kernelX)
    magY = cv2.filter2D(img, -1, kernelY)
    finalMag = np.ndarray((height,width))

    for i in range(height): 
        for j in range(width):
            finalMag[i][j] = int(math.sqrt((magX[i][j])**2 + (magY[i][j])**2))
    
    # plot mags
    plt.subplot(333),plt.imshow(magX, cmap='gray'),plt.title('X grad')
    plt.xticks([]), plt.yticks([])

    plt.subplot(334),plt.imshow(magY, cmap='gray'),plt.title('Y Grad')
    plt.xticks([]), plt.yticks([])

    # get our gradient directions
    directions = np.arctan(magY, magX)

    # we convert radians to degrees
    directions = (180/np.pi) * directions

    for i in range(height):
        for j in range(width):
            if(directions[i][j] < 0):
                # we get rid of all negative degrees 
                directions[i][j] = 180 + directions[i][j]

    return finalMag, directions

# Helper function to calculate the rounded angle values for non max suppressions
def roundAngles(angle): 
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle

""" This function is used as the first step to eliminate false positive edges 
    in our image. It checks the gradient direction at each pixel and then compares
    its intensity to the appropriate surrounding pixels"""
def NonMaxSuppression(img, directions, col, row): 

    # This is probably not needed but we use it anyway
    gradientDirections = directions
    
    for i in range(row):
        for j in range(col):
            gradientDirections[i][j] = roundAngles(directions[i][j])

    # do our direction check and supression
    for i in range(row):
        for j in range(col):
            # 0 checks left and right 
            if gradientDirections[i][j] == 0:
                # check the magnitude 
                if(j == 0):
                    # if we are on the left then only check the right 
                    if not img[i][j] > img[i][j+1]:
                        img[i][j] = 0
                if(j == col - 1):
                    # if we are on the right then only check the left
                    if not img[i][j] > img[i][j-1]:
                        img[i][j] = 0
                # we check both the left and the right 
                if not (img[i][j] > img[i][j-1] and img[i][j] > img[i][j+1]):
                    img[i][j] = 0
            
            # 45 checks up,right and down,left
            elif gradientDirections[i][j] == 45:
                # this checks to see if we are at the top left, bottom right of the images
                # and sets them to 0 b/c we can't do any comparison with this angle
                if((i == 0 and j == 0) or (i == row - 1 and j == col - 1)):
                    img[i][j] = 0
                # checks the bottom or the left 
                elif(i == row - 1 or j == 0):
                    # check up right only 
                    if not img[i][j] > img[i-1][j+1]:
                        img[i][j] = 0
                # checks top or right 
                elif(i == 0 or j == col - 1):
                    # check down left only 
                    if not img[i][j] > img[i+1][j-1]:
                        img[i][j] = 0
                else: 
                    # we can check both 
                    if not (img[i][j] > img[i+1][j-1] and img[i][j] > img[i-1][j+1]):
                        img[i][j] = 0
 
            # 90 checks up and down 
            elif gradientDirections[i][j] == 90:
                # check the magnitude 
                if i == 0:
                    # if we are on the top then only check the bottom 
                    if not img[i][j] > img[i+1][j]:
                        img[i][j] = 0
                elif i == row - 1:
                    # if we are on the bottom then only check the top
                    if not img[i][j] > img[i-1][j]:
                        img[i][j] = 0
                # we check both bottom and top
                elif not (img[i][j] > img[i-1][j] and img[i][j] > img[i+1][j]):
                    img[i][j] = 0
            
            # 135 checks up,left and down,right
            elif gradientDirections[i][j] == 135:
                # checks to see if we are top right or bottom left of the images 
                # and sets them to 0 b/c we can't do any comparison with this angle
                if((i == 0 and j == col - 1) or (i == row - 1 and j == 0)):
                    img[i][j] = 0
                # if we are the top or the left 
                elif(i == 0 or j == 0):
                    # check bottom right only 
                    if not img[i][j] > img[i+1][j+1]:
                        img[i][j] = 0
                # if we are bottom and right 
                elif(i == row -1 or j == col - 1):
                    # check top left only 
                    if not img[i][j] > img[i-1][j-1]:
                        img[i][j] = 0
                else:
                    # check both
                    if not (img[i][j] > img[i-1][j-1] and img[i][j] > img[i+1][j+1]):
                        img[i][j] = 0
    return img

""" This is the second part of our false positive edge detection where 
    we establish a lower and upper threshold to immediately decide which 
    intensities we want to keep.  Then we go through and edit any intensity 
    that might be an edge which is labeled as weak"""
def ThreshHold(img, col, row):

    # as per the slides our lower and upper bounds
    # 30 seems to work the best 
    lower = 30
    # lower = 20
    upper = lower * 1.5

    # our tested weak and strong values we set pixels to 
    weak = 50
    strong = 255

    # get indices for our strong edges 
    strongX, strongY = np.where(img > upper)

    # get indices for our weaker edges 
    weakX, weakY = np.where((img >= lower) & (img <= upper))

    # get indices for no edges 
    noX, noY = np.where(img < lower)

    # dummy array to hold our desired pixels 
    strengths = np.ndarray((row, col))

    # set our values with those indices we found earlier 
    strengths[strongX, strongY] = strong
    strengths[noX, noY] = 0
    strengths[weakX, weakY] = weak
    
    img[noX, noY] = 0

    # now we actually have to check if our weak edges are connected 
    # to any strong edges 
    for i in range(row):
        for j in range(col):
            
            if(img[i][j] == weak):
                # check to see if we are on the top row 
                if(i == 0):
                    # check the top left corner 
                    if(j == 0):
                        # we check right, bottom, and bottom right 
                        if(strengths[i][j+1] == strong or strengths[i+1][j] == strong or strengths[i+1][j+1] == strong):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
                    # check the top right corner
                    elif(j == col - 1):
                        # we check left, bottom, and bottom left 
                        if(strengths[i][j-1] == strong or strengths[i+1][j] == strong or strengths[i+1][j-1] == strong):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
                    # we are in the middle of the top row 
                    else: 
                        # we check right, left, bottom, bottom right, bottom left 
                        if(strengths[i][j-1] == strong or strengths[i][j+1] == strong or strengths[i+1][j] == strong or strengths[i+1][j+1] == strong or strengths[i+1][j-1] == strong):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
                # check to see if we are on bottom row
                elif(i == row - 1):
                    # check bottom left corner 
                    if(j == 0):
                        # we check right, above, and above right
                        if(strengths[i][j+1] == strong or strengths[i-1][j] == strong or strengths[i-1][j+1] == strong):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
                    # check bottom right corner 
                    elif(j == col - 1):
                        # we check left, above, above left
                        if(strengths[i][j-1] == strong or strengths[i-1][j] == strong or strengths[i-1][j-1] == strong):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
                    # in middle of the bottom row 
                    else: 
                        # we check right, left, above, above right, above left 
                        if(strengths[i][j-1] == strong or strengths[i][j+1] == strong or strengths[i-1][j] == strong or strengths[i-1][j+1] == strong or strengths[i-1][j-1] == strong):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
                # somewhere in the middle of the rows
                else:
                    # check the right
                    if(j == 0):
                        # we check above, below, right, above left, below left
                        if(strengths[i-1][j] == strong or strengths[i+1][j] == strong or strengths[i][j+1] == strong or strengths[i-1][j+1] == strong or strengths[i+1][j+1] == strong):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
                    # check the left 
                    elif(j == col - 1):
                        # we check above, below, left, above left, below left
                        if(strengths[i-1][j] == strong or strengths[i+1][j] == strong or strengths[i][j-1] == strong or strengths[i-1][j-1] == strong or strengths[i+1][j-1] == strong):
                            strengths[i][j] = strong
                        else:
                            strengths[i][j] = 0
                    else:
                        # we check above, below, right, left, above right, above left, below right, below left
                        if(strengths[i-1][j] == strong or strengths[i+1][j] == strong or strengths[i][j+1] == strong or strengths[i][j-1] == strong or strengths[i-1][j+1] == strong or strengths[i-1][j-1] == strong or strengths[i+1][j+1] == strong or strengths[i+1][j-1]):
                            strengths[i][j] = strong
                        else:
                            img[i][j] = 0
            
    return img

""" Our main caller function that handles the rest of the function calls"""
def CannyParent(name, col, row, sigma): 
    img = cv2.imread(name)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussImage = GaussianFilterParent(gray_image, sigma)
    
    gradientImage, directions = CentralGradient(gaussImage, col, row)

    plt.subplot(335),plt.imshow(gradientImage, cmap='gray'),plt.title('Final Magnitude')
    plt.xticks([]), plt.yticks([])

    suppressedPixels = NonMaxSuppression(gradientImage, directions, col, row)

    plt.subplot(337),plt.imshow(suppressedPixels, cmap='gray'),plt.title('Suppressed Pixels')
    plt.xticks([]), plt.yticks([])

    threshheldImage = ThreshHold(suppressedPixels, col, row)

    img = img.astype(int)
    plt.subplot(331),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])

    gaussImage = gaussImage.astype(int)
    plt.subplot(332),plt.imshow(gaussImage, cmap='gray'),plt.title('Gauss')
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(336),plt.imshow(directions, cmap='gray'),plt.title('Directions')
    plt.xticks([]), plt.yticks([])

    plt.subplot(338),plt.imshow(threshheldImage, cmap='gray'),plt.title('Threshhold Image')
    plt.xticks([]), plt.yticks([])
    plt.show()

def main(): 
    canny1Img = Image.open("./Images/canny1.jpg")
    canny2Img = Image.open("./Images/canny2.jpg") 
    image1 = Image.open("./Images/image1.png")
    image2 = Image.open("./Images/image2.png")

    col1, row1 = canny1Img.size
    col2, row2 = canny2Img.size
    colImg1, rowImg1 = image1.size
    colImg2, rowImg2 = image2.size
    # Sigma = 1 works best for this assignment as it blurs the image the least
    CannyParent("./Images/canny1.jpg", col1, row1, 1)
    CannyParent("./Images/canny2.jpg", col2, row2, 1)

    # Show effects of changing sigma 
    CannyParent("./Images/canny1.jpg", col1, row1, 3)
    CannyParent("./Images/canny2.jpg", col2, row2, 3)

    CannyParent("./Images/canny1.jpg", col1, row1, 5)
    CannyParent("./Images/canny2.jpg", col2, row2, 5)

    CannyParent("./Images/canny1.jpg", col1, row1, 10)
    CannyParent("./Images/canny2.jpg", col2, row2, 10)


    CannyParent("./Images/image1.png", colImg1, rowImg1, 1)
    CannyParent("./Images/image2.png", colImg2, rowImg2, 1)

main()