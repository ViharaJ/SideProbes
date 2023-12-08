"""
This script calculates the surface roughness using a Gaussian kernel. It processes 
a folder of images and returns a single average value for the stack. The input, 
'mainDir', is the directory which contains the images.
 
  
 
How to use (variables you need to change):
    1. Change variable mainDir to directory with folders of images
    2. Change csvOutputDir to directory where Surfaceroughness.xlsx will be saved
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
    4. Convert the XXX "recreated" exact contour to a Shapely object. For more info on Shapely
        see here: https://shapely.readthedocs.io/en/stable/geometry.html
    5. From the contour the algorithm recreates the baseline
    6. Iterate over the baseline (XXX find in each point the XXX slop XXX and XXX orthogonal XXX delete > euclidian distance), Turn 
    the normal line at each point to another Shapely object and find the intersection point XXX to the recreated contour XXX 
    7. Compute roughness XXX as euclidean distance between the baseline and created contour using the orthogonal XXX
 
    XXX ? Shortes distance > surface calculation? or recreation of the contour?
 
!!!Background Info!!!:
A CONTOUR is closed set of points. 
So, the contour of a straight line would include duplicate points. This is why 
we need to recreate the contour to include only unique points. Admittedly, this algorithm 
probably recreates an imperfect contour for specimens with re-entrant features so there is 
room for improvement.

XXX ? Can we please talk this through
XXX Shortest point of baseline to recreated contour
 
The sigma and kernel length for the Gauss kernel were found using a script. The
goal of this script was to create a baseline which closely matched the STL file of the 
speciment that was scanned. The code can be found here: https://github.com/ViharaJ/Find_Best_Kernel
 
The images are assumed to look similar to the top half of the outline of a circle. 
This is why the script searches for the lowest leftmost starting index. If your 
images are not suitable, you will have to change the code so that it finds the 
right starting point.

"""
import Module.Funcs_for_SR as fb
import numpy as np
import shapely
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import cv2
import os 
import sys
import math
import time
import pandas as pd
from scipy import spatial
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


def saveToExcel(porosity_data, names, rootDir, filename="Roughness"):
    df = pd.DataFrame(data=list(porosity_data), columns=['Roughness'], index=names)
    df.to_excel(os.path.join(rootDir, filename + ".xlsx"))


#===============================MAIN======================================
mainDir = "Z:\\Projekte\\42029-FOR5250\\Vihara\\Surface Roughness - All\\Roughness_Routine_Output\\Hantel01"
csvOutputDir= mainDir # CHANGE HERE

acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
dirPictures = os.listdir(mainDir)
scale = None
averageSR = []
Name = [mainDir.split("\\")[-1]] #get folder name


if(len(dirPictures)  <= 0):
    print('The specified folder is empty!')
    sys.exit()

else:
    counter = 0
    for path in dirPictures:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            
            
            # get scale from image
            if scale is None:
                scale = float(path.split("-")[1]) #find scale in the image
                
            #Ra
            distanceE = []
            
            # Extract contour
            img = cv2.imread(mainDir + '/' + path, cv2.IMREAD_GRAYSCALE)
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
                
                
                #Find nearest neighbour, stop when next vertex is dist > 15 away
                while(len(k) > 15):
                    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(k)
                    distance, indices = nbrs.kneighbors([newOrder[-1]])
                    
                    if(distance[0][0] > 15):
                        break
                    else:
                        indices = indices[:,0]
                        newOrder.append(k[indices[0]])
                        k = np.delete(k, indices[0], axis=0)
 
            
                #get unqiue points, maintain order (np.unique() doesn't inherently maintain order)
                _, idx = np.unique(newOrder, axis=0,  return_index=True)
                newOrderIndx = np.sort(idx)
                
                #array to store ordered points
                finalOrder = []
                
                for p in newOrderIndx:
                    finalOrder.append(newOrder[p])
                
                finalOrder = np.array(finalOrder)
          
                # recreated contour matches length criteria
                # XXX? original is divided by 2, since the contour of a line is a closed contour so every point is double
                # XXX? how do I know the above?
                if(len(finalOrder) >= (len(original)/2)*0.95): 
                    x = np.array(finalOrder[:,0])
                    y = np.array(finalOrder[:,1])
                    
                    #plot retrieved contour
                    ratio = img.shape[0]/img.shape[1]
                    plt.title(path)     
                    plt.plot(x, y, 'g.-', label="New contour")
                    
                    # get baseline
                    # XXX signal.convolve - Convolve (Falten) two N-dimensional arrays
                    # Faltung ist ein mathematischer Operator, der ein Signal mit einem anderen Signal mischt
                    xscipy = signal.convolve(x, kernel, mode='valid')
                    yscipy = signal.convolve(y, kernel, mode='valid')
                    
                    # dx & dy for the slope calculation
                    dx = np.diff(xscipy)
                    dy = np.diff(yscipy)
                    
                    #TODO REMOVE LATER;TESTING
                    print("Array lengths", len(x), len(xscipy))
                    
                    # plot baseline and show
                    plt.plot(xscipy, yscipy, 'm.-', label="baseline")
                    x_left, x_right = plt.gca().get_xlim()
                    y_low, y_high = plt.gca().get_ylim()
                    plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
                    plt.legend()
                    plt.show()
                    
                    # turn exact contour into shapely object
                    polyGon = shapely.geometry.LineString(finalOrder)
                    
                    
                    # iteratate over the baseline
                    for j in range(1,len(dx)):
                        # create normal line from point on the baseline
                        xs, ys = fb.createNormalLine(xscipy[j], yscipy[j], dx[j], dy[j])
                       
                        # turn normal line to shapely object
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
                            
                    
                    if len(distanceE) > 0:
                        print(np.average(distanceE)*scale*1000)
                        #average SR for image
                        averageSR.append(np.average(distanceE))
                        counter = counter + 1
            print(counter, "/", len(dirPictures))
                    
    
                
                
if len(averageSR) > 0:
    #averaage SR for whole folder
    print("Average Sa: ", np.average(averageSR)*scale*1000)
    
    # save to Excel 
    saveToExcel(np.average(averageSR)*scale*1000, Name, csvOutputDir)
    sys.exit()
