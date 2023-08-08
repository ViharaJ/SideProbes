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
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)

            if (cont):
                k = longestContour(cont)            
                # sig is sigma of Gauss, size is kernel's full length
                sig = 15
                size = 15
                distanceE = []     
                saveIndex = []
                
                x = np.array(k[:,0,0])*scale
                y = np.array(k[:,0,1])*scale
                

                # pairs = []                
                # for i in range(len(x)):
                #     if(doubleBack == 0 and [x[i], y[i]] in pairs):
                #         pairs = []
                #         pairs.append([x[i], y[i]])
                #         doubleBack = doubleBack + 1
                #     elif (doubleBack == 1 and [x[i], y[i]] in pairs):
                #         pairs = []
                #         pairs.append([x[i], y[i]])
                #         doubleBack = doubleBack + 1
                #  pairs.append([x[i], y[i]])
                    

                # rCont = np.squeeze(k*scale, axis=1)                                 
                # polyGon = shapely.geometry.LineString(rCont)
             
                sig = 15
                size = 12
                kernel = fb.gauss1D(size, sig)   
                
                xscipy = signal.convolve(x, kernel, mode='valid')
                yscipy = signal.convolve(y, kernel, mode='valid')
                
                dx = np.diff(xscipy)
                dy = np.diff(yscipy)
                
                plt.title(path)
                plt.plot(x,y, 'b.-', label="Exact contour")
                plt.plot(xscipy, yscipy, 'r.-', label="Baseline")
                plt.legend()
                plt.show()
                
                
                # for j in range(len(dx)):
                #     xs, ys = fb.createNormalLine(xscipy[j], yscipy[j], dx[j], dy[j])
                #     plt.plot(xscipy[j], yscipy[j], 'r.-')
                    
                #     stack = np.stack((xs,ys), axis=-1)
                #     line = shapely.geometry.LineString(stack)
                    
                #     #TODO remove this from main CODE
                #     if(polyGon.intersects(line) and j > 0):
                #         #intersection geometry
                #         interPoints = polyGon.intersection(line)
                        
                #         #intersection point
                #         mx, my = fb.proccessIntersectionPoint(interPoints, xscipy[j], yscipy[j])
                        
                #         euD = fb.euclidDist(xscipy[j], yscipy[j], mx, my)
                #         distanceE.append(euD)
                #         saveIndex.append(j)
                #         # plt.clf()
                #         # plt.plot(xscipy,yscipy, 'r.')
                #         # plt.plot(xs,ys, 'g-')
                #         # plt.plot(mx,my, 'mo')
                #         # plt.show()
                        
                
                # pairs = np.array(pairs)
                # plt.title(path)
                # plt.plot(pairs[:,0], pairs[:,1],'r.-')
                # plt.show()
                
                # if(keepImage(img, pairs, path) == False):
                #     removedImages.append(path)

