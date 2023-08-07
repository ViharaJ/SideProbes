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
                _, idx = np.unique(k, axis=0,  return_index=True)
                k = k[np.sort(idx, axis=-1)]
                k = np.squeeze(k, axis=1)
                
                minIndx = np.where(k[:,0] == k[:,0].min())[0][0]
                newOrder = [k[minIndx]]
                
                k = np.delete(k, minIndx, axis=0)
                
                
                while(len(k) > 1):
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(k)
                    distance, indices = nbrs.kneighbors([newOrder[-1]])
                    
                    indices = indices[:,0]
                    newOrder.append(k[indices[0]])
                    k = np.delete(k, indices[0], axis=0)
                    
                    
                
                x = []
                y = []
                
                for a,b in np.array(newOrder):
                    x.append(a)
                    y.append(b)
                    
                plt.title(path)
                plt.plot(x,y,'r.-')
                plt.show()
                
                
              
                        
                
  