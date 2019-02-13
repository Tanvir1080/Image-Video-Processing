from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Tanvir Islam 
# nid ta840968
# ucfid 3779383
# Spring 2019

"""horizontal axis is the pixel intensity, which can also be the number of bins if we change it from 256 
vertical axis is number of pixels at that intensity 
basically need to go through each pixel in the array and update frequency array at that pixel index with its intensity 

If we change the bin size, then the histogram itself should not actually change in its representation, that much. 
The only thing that changes is our x-axis which displays the number of bins.  
The trick is to get a singular bin to represent different amounts of intensities that will allow the shape to remain the same """
def Histogram(img, bins, col, row): 

    frequency = np.zeros(256)
    frequency = frequency.astype(int)
    widthArr = np.zeros(bins)
    widthArr = widthArr.astype(int)

    # we make our x-axis
    for i in range(bins):
        widthArr[i] = i

    # we loop through the image matrix and get our intensities as a regular array 
    for i in range(row):
        for j in range(col):
            frequency[img[i][j][0]] += 1
    
    binsFrequency = np.zeros(bins)
    numOfBins = 256 / bins
    numToBin = 0

    # here we use a third array to figure out how many intensities should be mapped 
    # to a single index in our x - axis (number of bins array)
    for i in range(256):

        # this allows us to skip over as many bins as we need and therefore we 
        # can combine multiple frequencies into a single bin if bins is less thatn 256
        if i % numOfBins == 0 and i is not 0:
            numToBin += 1
        binsFrequency[numToBin] += frequency[i]

    # we use our x - axis and new frequency array to finally plot the histogram
    plt.bar(widthArr, binsFrequency)
    plt.show()

def HistogramParent(img, binSize, col, row):
    arr = np.array(img)
    Histogram(arr, binSize, col, row)

def main(): 
    img4 = Image.open("image4.png")
    col, row = img4.size

    HistogramParent(img4, 256, col, row)
    HistogramParent(img4, 128, col, row)
    HistogramParent(img4, 64, col, row)

main()