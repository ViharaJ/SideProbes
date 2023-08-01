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


def longestContour(contours):
    maxIndx = 0
    maxLen = len(contours[0])
    
    for i in range(1, len(contours)):
        if (len(contours[i]) > maxLen):
            maxIndx = i
            maxLen = len(contours[i])
    
    return contours[maxIndx]


#"C:/Users/v.jayaweera/Pictures/FindingEdgesCutContour/OneFileContours"
sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Roughness_Routine_Output"
csvPath = '/Users/v.jayaweera/Documents/SRAvg-ContourDiv-NoInvert.csv'
acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
dirPictures = os.listdir(sourcePath)
imageID = []
scale = 16.12


if(len(dirPictures)  <= 0):
    print('The specified folder is empty!')
    sys.exit()
else:
    for path in dirPictures:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            #Reset plots to default figure size
            plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            
            # Extract MAIN contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            cont, hier = cv2.findContours(img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
            
            if (cont):
                k = longestContour(cont)            
                # sig is sigma of Gauss, size is kernel's full length
                sig = 15
                size = 15
                distanceE = []     
                saveIndex = []
                
                x = np.array(k[:,0,0])*scale
                y = np.array(k[:,0,1])*scale
                xscipy = []
                yscipy = []
                
                # plt.plot(x,y,'b.-',label='Exact contour')
                
                if(len(x) > size):
                    if(fb.euclidDist(x[0], y[0], x[-1], y[-1]) <= 150):
                        xscipy = scipy.ndimage.gaussian_filter(x, sig, radius=size, mode="wrap")
                        yscipy = scipy.ndimage.gaussian_filter(y, sig, radius=size, mode="wrap")
                    else:
                        xscipy = scipy.ndimage.gaussian_filter(x, sig, radius=size, mode="nearest")
                        yscipy = scipy.ndimage.gaussian_filter(y, sig, radius=size, mode="nearest")
                
                rCont = np.squeeze(k*scale, axis=1)                                 
                polyGon = shapely.geometry.LineString(rCont)
             
             
                dx = np.diff(xscipy)
                dy = np.diff(yscipy)
                
                
                for j in range(len(dx)):
                    xs, ys = fb.createNormalLine(xscipy[j], yscipy[j], dx[j], dy[j])
                    plt.plot(xscipy[j], yscipy[j], 'r.-')
                    
                    stack = np.stack((xs,ys), axis=-1)
                    line = shapely.geometry.LineString(stack)
                    
                    #TODO remove this from main CODE
                    if(polyGon.intersects(line) and j > 0):
                        #intersection geometry
                        interPoints = polyGon.intersection(line)
                        
                        #intersection point
                        mx, my = fb.proccessIntersectionPoint(interPoints, xscipy[j], yscipy[j])
                        
                        euD = fb.euclidDist(xscipy[j], yscipy[j], mx, my)
                        distanceE.append(euD)
                        saveIndex.append(j)
                        # plt.clf()
                        # plt.plot(xscipy,yscipy, 'r.')
                        # plt.plot(xs,ys, 'g-')
                        # plt.plot(mx,my, 'mo')
                        # plt.show()
                        
                # plt.gca().legend(('Exact contour','Baseline'))
                # plt.title(path)
                # plt.show()
                
                
    shiftedDistance = np.array(distanceE)-np.average(distanceE)
    print("Surface Roughness ", path, ': ', np.average(abs(shiftedDistance)))