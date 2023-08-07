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

<<<<<<< HEAD
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
=======

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

>>>>>>> doubleBack

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
<<<<<<< HEAD
            doubleBack = 0
            #Reset plots to default figure size
            plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            plt.gca().invert_yaxis()
            
            
=======
>>>>>>> doubleBack
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)

            if (cont):
                #Get main contour of interest, ignore pores
                k = longestContour(cont)
<<<<<<< HEAD
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
                
                
              
                        
                
  
=======
                                         
                x = np.array(k[:,0,0])*scale
                y = np.array(k[:,0,1])*scale
                
                pairs = []
                
                for i in range(len(x)):
                    if(doubleBack == 0 and [x[i], y[i]] in pairs):
                        pairs = []
                        pairs.append([x[i], y[i]])
                        doubleBack = doubleBack + 1
                    elif (doubleBack == 1 and [x[i], y[i]] in pairs):
                        pairs = []
                        pairs.append([x[i], y[i]])
                        doubleBack = doubleBack + 1
                
                    pairs.append([x[i], y[i]])
                    
                
                # pairs = np.array(pairs)
                # plt.title(path)
                # plt.plot(pairs[:,0], pairs[:,1],'r.-')
                # plt.show()
                
                if(keepImage(img, pairs, path) == False):
                    removedImages.append(path)

>>>>>>> doubleBack
