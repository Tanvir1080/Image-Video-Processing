from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# Tanvir Islam 
# nid ta840968
# ucfid 3779383
# Spring 2019

def CentralGradient(img, width, height):
    kernelX = np.array([[-1,0,1], [-1, 0, 1], [-1, 0, 1]]) / 3
    kernelY = np.array([[1,1,1], [0,0,0], [-1, -1, -1]]) / 3
    
    magX = cv2.filter2D(img, -1, kernelX)
    magY = cv2.filter2D(img, -1, kernelY)
    finalMag = np.ndarray((height,width, 3))
    # finalMag = finalMag.astype(int)

    for i in range(height): 
        for j in range(width):
            for k in range(3):
                finalMag[i][j][k] = int(math.sqrt((magX[i][j][k])**2 + (magY[i][j][k])**2))

    finalMag = finalMag.astype(int)
    plt.subplot(231),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(magX),plt.title('Central: X Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(magY),plt.title('Central: Y Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(finalMag),plt.title('Central: Final Gradient')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def BackwardGradient(img, width, height):
    kernelX = np.array([-1,1]) / 2
    kernelY = np.ndarray((2,1)) / 2
    kernelY[0][0] = -1
    kernelY[1][0] = 1

    magX = cv2.filter2D(img, -1, kernelX)
    magY = cv2.filter2D(img, -1, kernelY)
    finalMag = np.ndarray((height,width, 3))

    for i in range(height): 
        for j in range(width):
            for k in range(3):
                finalMag[i][j][k] = int(math.sqrt((magX[i][j][k])**2 + (magY[i][j][k])**2))
    
    finalMag = finalMag.astype(int)
    plt.subplot(231),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(magX),plt.title('Backward: X Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(magY),plt.title('Backward: Y Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(finalMag),plt.title('Backward: Final Gradient')
    plt.xticks([]), plt.yticks([])
    plt.show()

def ForwardGradient(img, width, height):
    kernelX = np.array([1,-1]) / 2
    kernelY = np.ndarray((2,1)) / 2
    kernelY[0][0] = 1
    kernelY[1][0] = -1

    magX = cv2.filter2D(img, -1, kernelX)
    magY = cv2.filter2D(img, -1, kernelY)
    finalMag = np.ndarray((height,width, 3))

    for i in range(height): 
        for j in range(width):
            for k in range(3):
                finalMag[i][j][k] = int(math.sqrt((magX[i][j][k])**2 + (magY[i][j][k])**2))
    
    finalMag = finalMag.astype(int)
    plt.subplot(231),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(magX),plt.title('Forward: X Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(magY),plt.title('Forward: Y Gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(finalMag),plt.title('Forward: Final Gradient')
    plt.xticks([]), plt.yticks([])
    plt.show()

def main():
    img = Image.open("./Images/image3.png")
    img3 = cv2.imread("./Images/image3.png")
    col, row = img.size
    CentralGradient(img3, col, row)
    BackwardGradient(img3, col, row)
    ForwardGradient(img3, col, row)

main()