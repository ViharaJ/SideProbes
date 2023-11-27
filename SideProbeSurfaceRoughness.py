"""
This script calculates the surface roughness using a Gaussian kernel. It processes 
a folder of images and returns a single average value for the stack. The input, 
'sourcePath', is the directory which contains folders of images.

How to use: 
    1. Change sourcePath
    2. Change csvOutputPath
    3. Ensure that the image names have the scale in the second position.
        Everything should be seperated by '-'


How it works:
    1. Find the longest contour in image
    2. Finds the lowest leftmost pixel in the image
    3. Recreate contour such that duplicate points are removed
        Currently, if the next closes vertex to the current point is more than
        5 pixels away, we stop recreating the contour and break out of the routine
        If the new contour is about 95% the contour (unique vertices only), compute the roughness
            -Note: These values 5 pixels and 95% were chosen arbitrarily
            
    4. Convert the exact contour to a Shapely object. For more info on Shapely
        see here: https://shapely.readthedocs.io/en/stable/geometry.html
    5. Iterate over the baseline
    6. Turn the normal line at each point to another Shapely object and find the intersection point
    7. Compute roughness


Contour recreating method: Find nearest neighbour, then keep unqiue points



!!!Background Info!!!:
    
A CONTOUR is closed set of points. 
So, the contour of a straight line would include duplicate points. This is why 
we need to recreate the contour to include only unique points. 

The sigma and kernel length for the Gauss kernel were found using a script. The
goal of this script was to create a baseline which closely matched the STL file of the 
speciment that was scanned. The code can be found here: https://github.com/ViharaJ/Find_Best_Kernel

The images are assumed to look similar to the top half of the outline of a circle. 
This is why the script searches for the lowest leftmost starting index. If your 
images are not suitable, you will have to change the code so that it finds the 
right starting point. 

"""

import numpy as np
import shapely
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import Module.Functions as fb # functions from other module with a collection of functions
import os 
import sys
import time
from sklearn.neighbors import NearestNeighbors

#==================FUNCTIONS======================================
def longestContour(contours):
    """
    contours: contours returned by cv2
    returns: longest contour input list
    """
    maxIndx = 0
    maxLen = len(contours[0])
    
    for i in range(1, len(contours)):
        if (len(contours[i]) > maxLen):
            maxIndx = i
            maxLen = len(contours[i])
    
    return contours[maxIndx]


def nearestNeighbour(x1, y1, allX, allY):
    """
    Find the nearest point to (x1,y1) from arrays
    allX and allY.
    """
    distance = fb.euclidDist(x1, y1, allX, allY)   
    return np.where(distance == np.min(distance))[0]


def recreateContour(fullContour):
    """
    fullContour: contour of shape (n,2), this should be the contour returned from 
        cv2.findContours
    returns: new contour of shape (n,2) 
    """
    #find starting point of contour
    minIndices = np.where(fullContour[:,1] == fullContour[:,1].max())[0]
    minPoints = fullContour[minIndices]
    minIndx = np.where(minPoints[:,0] == minPoints[:,0].min())[0][0]
    startingCord = fullContour[minIndices[minIndx]]
    
    #array to store ordered points
    newOrder = [startingCord]
    
    #delete starting point from contour array (only pairs values in k)
    fullContour = np.delete(fullContour, minIndx, axis=0)
    
    
    #Find nearest neighbour, stop when next vertex is dist > 15 pixels away
    while(len(fullContour) > 1):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(fullContour)
        distance, indices = nbrs.kneighbors([newOrder[-1]])
        
        if(distance[0][0] > 15): # if next verteix is 15 pixels away, brek
            break
        else:
            indices = indices[:,0]
            newOrder.append(fullContour[indices[0]])
            fullContour = np.delete(fullContour, indices[0], axis=0)


    #get unqiue points, maintain order
    _, idx = np.unique(newOrder, axis=0,  return_index=True)
    newOrderIndx = np.sort(idx)
    
    finalOrder = []
    
    for p in newOrderIndx:
        finalOrder.append(newOrder[p])
    
    return np.array(finalOrder)
    
#===============================MAIN======================================
start = time.time()
sourcePath = "C:\\Users\\v.jayaweera\\Pictures\\Probe01ROI2"
csvOutputPath = '/Users/v.jayaweera/Documents/Hantel03_Try3_Outline_Filtered-SRAvg.csv'

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
    for path in dirPictures:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            
            # get scale from image
            if scale is None:
                scale = float(path.split("-")[1])
            
            # Ra, image used
            distanceE, saveIndex = [], []
            
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)

            if (cont):
                #Get main contour of interest, ignore pores
                k = longestContour(cont)
                
                #turn contour to array shape (n,2)
                k = np.squeeze(k, axis=1)
                original = k
                
                #plot original contours
                # plt.plot(k[:,0], k[:,1],'r.-', label="Exact contour")
                
                # get recreated contour
                finalOrder = recreateContour(k)
                
                # sig is sigma of Gauss, size is kernel's full length
                # create Gauss kernel
                sig = 350
                size = 319
                kernel = fb.gauss1D(size, sig)   
          
                # recreated contour matches length criteria
                if(len(finalOrder) >= (len(original)/2)*0.95): 
                    x = np.array(finalOrder[:,0])
                    y = np.array(finalOrder[:,1])
                    
                    # plot recreated contour
                    ratio = img.shape[0]/img.shape[1]
                    plt.title(path)     
                    plt.plot(x, y, 'g.-', label="New contour")
                    
                    # get baseline
                    xscipy = signal.convolve(x, kernel, mode='valid')
                    yscipy = signal.convolve(y, kernel, mode='valid')
                    
                    dx = np.diff(xscipy)
                    dy = np.diff(yscipy)
                    
                    # TODO REMOVE LATER;TESTING
                    print("Array lengths", len(x), len(xscipy))
                    
                    # plot baseline and show
                    plt.plot(xscipy, yscipy, 'm.-', label="baseline")
                    x_left, x_right = plt.gca().get_xlim()
                    y_low, y_high = plt.gca().get_ylim()
                    plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
                    plt.legend()
                    plt.show()
                    
                    # turn contour to shapely object
                    polyGon = shapely.geometry.LineString(finalOrder)
                    
                    # iteratate over the baseline
                    for j in range(1,len(dx)):
                        # create normal line from point on the baseline
                        xs, ys = fb.createNormalLine(xscipy[j], yscipy[j], dx[j], dy[j])
                        
                        # turn normal line to shapely object
                        stack = np.stack((xs,ys), axis=-1)
                        line = shapely.geometry.LineString(stack)
                        
                        
                        if (polyGon.intersects(line)):
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

