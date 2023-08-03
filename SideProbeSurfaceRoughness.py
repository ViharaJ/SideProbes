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

def longestContour(contours):
    maxIndx = 0
    maxLen = len(contours[0])
    
    for i in range(1, len(contours)):
        if (len(contours[i]) > maxLen):
            maxIndx = i
            maxLen = len(contours[i])
    
    return contours[maxIndx]


start = time.time()
#"C:/Users/v.jayaweera/Pictures/FindingEdgesCutContour/OneFileContours"
sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Roughness_Routine_Output/Hantel01_Outline"
csvPath = '/Users/v.jayaweera/Documents/SRAvg-ContourDiv-NoInvert.csv'
acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
dirPictures = os.listdir(sourcePath)
imageID = []
scale = 6.249
averageSR = []

if(len(dirPictures)  <= 0):
    print('The specified folder is empty!')
    sys.exit()
else:
    
    for path in dirPictures:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            print(path)
            #Reset plots to default figure size
            plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            plt.gca().invert_yaxis()
            
            
            
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
            ratio = img.shape[0]/img.shape[1]
           
           
            
            if (cont):
                #Get main contour of interest, ignore pores
                k = longestContour(cont)
                #REMOVE properly
                k = np.unique(k, axis=0)
                
                # sig is sigma of Gauss, size is kernel's full length
                sig = 60
                size = 20
                distanceE = []   
                saveIndex = []
                            
                x = np.array(k[:,0,0])*scale
                y = np.array(k[:,0,1])*scale
                xscipy = []
                yscipy = []
                gauss_kernel = fb.gauss1D(size, sig)
               
                # pointsThusFarX = []
                # pointsThusFarY = []
                # for d in range(len(x)):
                #     plt.title(time.time()-start)
                #     plt.plot(pointsThusFarX, pointsThusFarY, 'b.-')
                #     plt.plot(x[d], y[d], 'r.')
                #     plt.show()
                #     pointsThusFarX.append(x[d])
                #     pointsThusFarY.append(y[d])
                
                #Get baseline
                if(len(x) > size):
                    if(fb.euclidDist(x[0], y[0], x[-1], y[-1]) <= 10):
                       xscipy = scipy.signal.convolve(x, gauss_kernel, mode="valid")
                       yscipy = scipy.signal.convolve(y, gauss_kernel, mode="valid")
                    else:
                        print("valid")
                        xscipy = scipy.signal.convolve(x, gauss_kernel, mode="valid")
                        yscipy = scipy.signal.convolve(y, gauss_kernel, mode="valid")
                        
                plt.plot(x,y,'b.-',label='Exact contour')
                plt.plot(xscipy, yscipy, 'r.-', label='Baseline')
                
                #Convert to Shapely geometry object
                rCont = np.squeeze(k*scale, axis=1)                                 
                polyGon = shapely.geometry.LineString(rCont)
                
                #derivative of baseline
                dx = np.diff(xscipy)
                dy = np.diff(yscipy)
                
                #TODO: Check change
                for j in range(1, len(dx)):
                    xs, ys = fb.createNormalLine(xscipy[j], yscipy[j], dx[j], dy[j])
                    
                    #Convert ortho line to Shapely geometry object
                    stack = np.stack((xs,ys), axis=-1)
                    line = shapely.geometry.LineString(stack)
                    
                    
                    if(polyGon.intersects(line) and j > 0):
                        #intersection geometry
                        interPoints = polyGon.intersection(line)
                        
                        #intersection point
                        mx, my = fb.proccessIntersectionPoint(interPoints, xscipy[j], yscipy[j])
                        # plt.plot(x,y,'b.-',label='Exact contour')
                        # plt.plot(xscipy, yscipy, 'r.-', label='Baseline')
                        # plt.plot( mx, my, 'y.-')
                        # plt.show()
                        
                        euD = fb.euclidDist(xscipy[j], yscipy[j], mx, my)
                        distanceE.append(euD)
                        saveIndex.append(j)
                       
                
            #get x and y limits
            x_left, x_right = plt.gca().get_xlim()
            y_low, y_high = plt.gca().get_ylim()
            plt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
            plt.legend()
            plt.title(path)
            plt.show()
            print(np.average(distanceE))
            print(np.average(abs(np.array(distanceE)-np.average(distanceE))))
            
            mean = np.average(distanceE)
            averageSR.append(np.average(abs(np.array(distanceE)-mean)))
            
             
            #Euclid distance 
            xPos = [0]
            sumDist = 0
             
            for j in range(len(saveIndex)):
                n = saveIndex[j]
                temp = fb.euclidDist(xscipy[n-1], yscipy[n-1], xscipy[n], yscipy[n])
                sumDist = sumDist + temp
                xPos.append(sumDist)
            
                        
            plt.rcParams["figure.figsize"] = (12,3)
            plt.title('Projection')
            plt.ylabel('Ortho. Distance')
            plt.xlabel('Distance between points')
            plt.plot(xPos[:-1], np.array(distanceE)-mean, 'm.-')
            plt.axhline(y=0, color='b', linestyle='-', label='mean')
            plt.xticks(np.arange(min(xPos), max(xPos)+1, 10.0))
            plt.xlim([0,190])
            plt.legend()
            plt.show()
            
            if len(averageSR) > 10:
                break
            
    print("Surface Roughness ", path, ': ', np.average(averageSR))