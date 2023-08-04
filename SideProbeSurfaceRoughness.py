# -*- coding: utf-8 -*-
"""
Input: Binary image(s)
Scale: adjust to fit image resolution
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

if(len(dirPictures)  <= 0):
    print('The specified folder is empty!')
    sys.exit()
else:
    
    for path in dirPictures:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            doubleBack = 0
            #Reset plots to default figure size
            plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            plt.gca().invert_yaxis()
            
            
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
            cont, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            if (cont):
                #Get main contour of interest, ignore pores
                k = longestContour(cont)
                #REMOVE properly
                # _, idx = np.unique(k, axis=0,  return_index=True)
                # k = k[np.sort(idx, axis=-1)]
                
                x = k[:,0,0]                
                y = k[:,0,1]
                
                dataset = []
                for j in range(len(x)):
                    dataset.append([x[j],y[j]])
               
                dataset.sort(key=lambda x:x[0])


                #spatially organising the points on a tree for quick nearest neighbors calc
                kdtree = spatial.KDTree(dataset)
                
                #calculates the nearest neighbors of each point
                _ , neighs = kdtree.query(dataset, k=5)
                
               
                newSet = []
                newSet.append(dataset[0])
                dataset = np.delete(dataset, 0, axis=0)
                
                
                while(len(dataset) > 0):
                    coord = newSet[-1]
                    #calculates the nearest neighbors of each point
                    _ , neighs = kdtree.query([coord], k=5)
                    _, idx = np.unique(neighs, axis=0,  return_index=True)
                    neighs = neighs[np.sort(idx, axis=-1)]
        
                    dataset = np.delete(dataset, neighs[0], axis=0)
                    newSet.append(dataset[neighs[0]])
                        
                
                
# Manual nearest neighbour
# if (cont):
#     #Get main contour of interest, ignore pores
#     k = longestContour(cont)
#     #REMOVE properly
#     # _, idx = np.unique(k, axis=0,  return_index=True)
#     # k = k[np.sort(idx, axis=-1)]
    
#     x = k[:,0,0]                
#     y = k[:,0,1]
    
#     dataset = []
#     for j in range(len(x)):
#         dataset.append([x[j],y[j]])
    
#     dataset.sort(key=lambda x:x[0])

    
#     newSet = []
#     newSet.append(dataset[0])
#     dataset = np.delete(dataset, 0, axis=0)
    
    
#     while(len(dataset) > 0):
#         coord = newSet[-1]
#         allX = dataset[:,0]
#         allY = dataset[:,1]
#         swapCoord = nearestNeighbour(coord[0], coord[1], allX, allY)
        
#         newSet.append(dataset[swapCoord[0]])
#         dataset = np.delete(dataset, swapCoord, axis=0)
        
        
    
#     newSet = np.array(newSet)
#     print(newSet)
#     plt.plot(newSet[:,0], newSet[:,1], 'r.-')
#     plt.show()