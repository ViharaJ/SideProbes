# -*- coding: utf-8 -*-
"""
kNN2: Find nearest neighbour, then keep unqiue points
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

# '''
# radius = nonminal file;
# fit circle
# '''
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
# sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Temporary Scripts/CreateRemoval_CSV_Doc/Hantel01_Filtered"
sourcePath = "C:/Users/v.jayaweera/Documents/Anne/Side Probes/Roughness_Routine_Output_Downskin/Hantel16-C1"
csvPath = '/Users/v.jayaweera/Documents/Hantel03_Try3_Outline_Filtered-SRAvg.csv'

acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
dirPictures = os.listdir(sourcePath)
imageID = []
scale = None
averageSR = []



if(len(dirPictures)  <= 0):
    print('The specified folder is empty!')
    sys.exit()

else:
    counter = 0
    for path in dirPictures[::2]:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            
            if scale is None:
                scale = float(path.split("-")[1])
                
            distanceE = []
            saveIndex = []
            
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)

            if (cont):
                #Get main contour of interest, ignore pores
                k = longestContour(cont)
                
                #turn contour to shape (n,2)
                k = np.squeeze(k, axis=1)
                original = k 
                #plot original contours
                # plt.plot(k[:,0], k[:,1],'r.-', label="Exact contour")
                
                # sig is sigma of Gauss, size is kernel's full length
                sig = 350
                size = 319
                kernel = fb.gauss1D(size, sig)   
            
                #find starting point of contour
                minIndices = np.where(k[:,1] == k[:,1].max())[0]
                minPoints = k[minIndices]
                minIndx = np.where(minPoints[:,0] == minPoints[:,0].min())[0][0]
                startingCord = k[minIndices[minIndx]]
                
                #array to store ordered points
                newOrder = [startingCord]
                
                #delete starting point from contour array (only pairs values in k)
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
 
            
                #get unqiue points, maintain order
                _, idx = np.unique(newOrder, axis=0,  return_index=True)
                newOrderIndx = np.sort(idx)
                
                finalOrder = []
                
                for p in newOrderIndx:
                    finalOrder.append(newOrder[p])
                
                finalOrder = np.array(finalOrder)
          
                if(len(finalOrder) >= (len(original)/2)*0.50): 
                    x = np.array(finalOrder[:,0])
                    y = np.array(finalOrder[:,1])
                    
                    #plot retrieved contour
                    ratio = img.shape[0]/img.shape[1]
                    plt.title(path)     
                    plt.plot(x, y, 'g.-', label="New contour")
                    
                    #get baseline
                    xscipy = signal.convolve(x, kernel, mode='valid')
                    yscipy = signal.convolve(y, kernel, mode='valid')
                    
                    dx = np.diff(xscipy)
                    dy = np.diff(yscipy)
                    
                    #TODO REMOVE LATER;TESTING
                    print("Array lengths", len(x), len(xscipy))
                    
                    plt.plot(xscipy, yscipy, 'm.-', label="baseline")
                    x_left, x_right = plt.gca().get_xlim()
                    y_low, y_high = plt.gca().get_ylim()
                    plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
                    plt.legend()
                    plt.show()
                    
                    polyGon = shapely.geometry.LineString(finalOrder)
                    
                    for j in range(1,len(dx)):
                        xs, ys = fb.createNormalLine(xscipy[j], yscipy[j], dx[j], dy[j])
                       
                        
                        stack = np.stack((xs,ys), axis=-1)
                        line = shapely.geometry.LineString(stack)
                        
                        #TODO remove this from main CODE
                        if(polyGon.intersects(line)):
                            #intersection geometry
                            interPoints = polyGon.intersection(line)
                            
                            #intersection point
                            mx, my = fb.proccessIntersectionPoint(interPoints, xscipy[j], yscipy[j])
                            
                            euD = fb.euclidDist(xscipy[j], yscipy[j], mx, my)
                            distanceE.append(euD)
                            saveIndex.append(j)
                    
                    if len(distanceE) > 0:
                        print(np.average(distanceE))
                        averageSR.append(np.average(distanceE))
                        counter = counter + 1
            print(counter, "/", len(dirPictures))
                
                
if len(averageSR) > 0:
    print("Average Sa: ", np.average(averageSR)*scale*1000)
    sys.exit()

