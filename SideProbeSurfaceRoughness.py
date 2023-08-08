# -*- coding: utf-8 -*-
"""
kNN on unique points
"""
import numpy as np
import shapely
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import cv2
import Module.Functions as fb
import os 
import sys
import math
import time
from scipy import spatial
from sklearn.neighbors import NearestNeighbors

def longestContour(contours):
    maxIndx = 0
    maxLen = len(contours[0])
    
    for i in range(1, len(contours)):
        if (len(contours[i]) > maxLen):
            maxIndx = i
            maxLen = len(contours[i])
    
    return contours[maxIndx]


def checkFeature(image, row, col):
    image = np.array(image)
    
    loops = 5
    #check back
    back = col
    while back > 0 and loops >= 0:        
        prevCol = img[:, col-1]        
        whitePx = np.where(prevCol==255)[0]
        
        if len(whitePx) > 1:
            return True
        back = back - 1
        loops = loops - 1
        
    #check forward
    loops = 5
    forward = col
    while forward < image.shape[1] and loops >= 0:        
        prevCol = img[:, col+1]        
        whitePx = np.where(prevCol==255)[0]
        
        if len(whitePx) > 1:
            return True
        forward = forward + 1
        loops = loops - 1
        
    return False

def keepImage(image, contour, filename):
    """
    img: img opencv array
    """
    contour = np.array(contour)
    ratio = image.shape[0]/image.shape[1]
    fig, axs = plt.subplots(2, 1)
    fig.suptitle(filename)
    axs[0].imshow(image)
    axs[1].invert_yaxis()
    axs[1].plot(contour[:,0], contour[:,1], 'b.-')
    x_left, x_right = plt.gca().get_xlim()
    y_low, y_high = plt.gca().get_ylim()
    fig.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.show()
    
    inp = input("Keep (1 or SPACE) or Remove(2)?")
   
    if(inp == '' or inp == '1'):
        return True
    else:
        return False
    
def makeComparisonPlot(image, x,y, bx,by):
    #Reset plots to default figure size
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.gca().invert_yaxis()
    ratio = image.shape[0]/image.shape[1]
    
    #plot
    plt.plot(x,y,'b.-',label='Exact contour')
    plt.plot(bx, by, 'r.-', label='Baseline')
    
    #get x and y limits and resize axes
    x_left, x_right = plt.gca().get_xlim()
    y_low, y_high = plt.gca().get_ylim()
    plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    plt.legend()
    plt.title(path)
    plt.show()



def nearestNeighbour(x1, y1, allX, allY):
    distance = fb.euclidDist(x1, y1, allX, allY)   
    return np.where(distance == np.min(distance))[0]

    

start = time.time()
#"C:/Users/v.jayaweera/Pictures/FindingEdgesCutContour/OneFileContours"
sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Roughness_Routine_Output/Hantel01_Outline"
csvPath = '/Users/v.jayaweera/Documents/SRAvg-ContourDiv-NoInvert.csv'
acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
dirPictures = os.listdir(sourcePath)
imageID = []
scale = 6.249
averageSR = []
doubleBack = 0
removedImages = []


if(len(dirPictures)  <= 0):
    print('The specified folder is empty!')
    sys.exit()
else:    
    for path in dirPictures:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            doubleBack = 0
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)

            if (cont):
                #Get main contour of interest, ignore pores
                k = longestContour(cont)
                
                #plot original contour
                plt.plot(k[:,0,0],k[:,0,1],'r.-')
                
                #get unqiue points, maintain order
                _, idx = np.unique(k, axis=0,  return_index=True)
                k = k[np.sort(idx, axis=-1)]
                
                #turn contour to shape (n,2)
                k = np.squeeze(k, axis=1)
                
                #find starting point of contour
                minIndices = np.where(k[:,1] == k[:,1].min())[0]
                minPoints = k[minIndices]
                minIndx = np.where(minPoints[:,0] == minPoints[:,0].min())[0][0]
                startingCord = k[minIndices[minIndx]]
                
                #array to store ordered points
                newOrder = [startingCord]
                
                #delete starting point from contour array (only unique pairs in k)
                k = np.delete(k, minIndx, axis=0)
                
                
                #Find nearest neighbour, stop when next vertex is dist > 4 away
                while(len(k) > 1):
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(k)
                    distance, indices = nbrs.kneighbors([newOrder[-1]])
                    
                    if(distance[0][0] > 4):
                        break
                    else:
                        indices = indices[:,0]
                        newOrder.append(k[indices[0]])
                        k = np.delete(k, indices[0], axis=0)
 
                #turn newOrder to array
                newOrder = np.array(newOrder)
                
                #plot retrieved contour
                plt.title(path)               
                plt.plot(newOrder[:,0], newOrder[:,1], 'g.-')
                plt.show()
                    
        

