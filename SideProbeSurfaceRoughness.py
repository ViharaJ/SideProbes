"""
This script calculates the surface roughness using a Gaussian kernel. It processes 
a folder of images and returns a single average value for the stack. The input, 
'mainDir', is the directory which contains folders of images.
 
The folders specified in subFolders_Of_Interest will be processed, a single surface
roughness value will be returned for each folder. Upon completion, the results 
for each folder will be saved as an excel file
 
 
 
How to use (variables you need change):
    1. Change variable mainDir to directory with folders of iamges
    2. Update subFolders_Of_Interest to include folders you want to process
    2. Change csvOutputDir to directory where Porosity.xlsx will be saved
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
    5. From the contour the algorithm recreates the baseline
    6. Iterate over the baseline (slop, orthogonal, euclidian distance), Turn 
    the normal line at each point to another Shapely object and find the intersection point
    7. Compute roughness
 
 
!!!Background Info!!!:
A CONTOUR is closed set of points. 
So, the contour of a straight line would include duplicate points. This is why 
we need to recreate the contour to include only unique points. Admittedly, this algorithm 
probably recreates an imperfect contour for specimens with re-entrant features so there is 
room for improvement.
 
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
import Module.Funcs_for_SR as fb # functions from another script with a collection of functions
import os 
import sys
from sklearn.neighbors import NearestNeighbors
import pandas as pd

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

    
def saveToExcel(porosity_data, names, rootDir, filename="Roughness"):
    df = pd.DataFrame(data=list(porosity_data), columns=['Roughness'], index=names)
    df.to_excel(os.path.join(rootDir, filename + ".xlsx")) 
    

def calculateSR(img, scale, sigma, kernel_len):
    '''
    img: image to be processed
    s: sigma for gaussl kernel
    k: full kernel length
    returns: Surface roughness or -1 if failed to find a value
    '''
    # Ra
    distanceE = []
    #extract contour
    cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
    
    if cont is None:
        return -1
    
    #Get main contour of interest, ignore pores
    k = longestContour(cont)
    
    #turn contour to array shape (n,2)
    k = np.squeeze(k, axis=1)
    original = k
    
    # get recreated contour
    finalOrder = recreateContour(k)
    
    # create Gauss kernel
    kernel = fb.gauss1D(sigma, kernel_len)   
  
    # recreated contour matches length criteria
    if(len(finalOrder) >= (len(original)/2)*0.95): 
        x = np.array(finalOrder[:,0])
        y = np.array(finalOrder[:,1])
        
        # plot recreated contour
        ratio = img.shape[0]/img.shape[1]
        # plt.title(path)     
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
            
    #TODO: CHECK WHY NO INTERSECTIONS ARE FOUND
    return -1 if len(distanceE) == 0 else np.average(distanceE)
    
#===============================MAIN======================================
mainDir = "Z:\\Projekte\\42029-FOR5250\\Vihara\\Test-gyroid\\probe 4\\processed images"

subFolders_Of_Interest =["Outer_Surface_Downskin","Outer_Surface_SideSkin_Left","Outer_Surface_SideSkin_Right", "Outer_Surface_Upskin"]
csvOutputDir = "Z:\\Projekte\\42029-FOR5250\\Vihara\\Test-gyroid\\probe 4\\processed images\\Documents"

acceptedFileTypes = ["jpg", "png", "bmp", "tif"]

scale = None
averageSR = []
names = []


for folder in os.listdir(mainDir):
    # get full folder path
    f_path = os.path.join(mainDir, folder)
    
    # check if it's both a folder and a folder we're processing
    SR_of_folder = []
    if os.path.isdir(f_path) and folder in subFolders_Of_Interest:       
        names.append(folder)
        print("Processing: ",  f_path)
        
        counter = 0
        for image_name in os.listdir(f_path):
            if( '.' in image_name and image_name.split('.')[-1].lower()
               in acceptedFileTypes):
                img = cv2.imread(os.path.join(f_path, image_name), cv2.IMREAD_GRAYSCALE)
                
                # get scale from image
                if scale is None:
                    scale = float(image_name.split("-")[1])
                
                
                SR = calculateSR(img, scale, 350, 319)
                
                if SR != -1:
                    SR_of_folder.append(SR)                            
                    #Number of images used for final calculations
                    counter = counter + 1
                
                print(counter, "/", len(os.listdir(f_path)))
    
    
        if len(SR_of_folder) > 0:
            averageSR.append(np.average(SR_of_folder))
    
saveToExcel(averageSR, names, csvOutputDir)


        


