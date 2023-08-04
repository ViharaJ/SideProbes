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
sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Roughness_Routine_Output/HantelTest"
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
                print(idx.shape)
                k = k[np.sort(idx)]
                
                # sig is sigma of Gauss, size is kernel's full length
                sig = 60
                size = 20
                distanceE = []   
                saveIndex = []
                            
                x = np.array(k[:,0,0])*scale
                y = np.array(k[:,0,1])*scale
                
                ppx = []
                ppy = []
                
                for i in range(len(x)):
                    plt.plot(ppx, ppy, 'b.-')
                    plt.plot(x[i], y[i], 'r.')
                    plt.show()
                    ppx.append(x[i])
                    ppy.append(y[i])
            
                # plt.plot(x,y,'r.-')
                # plt.show()
                # sys.exit()
                
                
                
####testing using svg                
# from scipy import signal
# import scipy
# import cv2
# import Module.Functions as fb
# import os 
# import sys
# import math
# import time
# import potrace
# import numpy as np
# import matplotlib.pyplot as plt

# def longestContour(contours):
#     maxIndx = 0
#     maxLen = len(contours[0])
    
#     for i in range(1, len(contours)):
#         if (len(contours[i]) > maxLen):
#             maxIndx = i
#             maxLen = len(contours[i])
    
#     return contours[maxIndx]


# start = time.time()
# #"C:/Users/v.jayaweera/Pictures/FindingEdgesCutContour/OneFileContours"
# sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Roughness_Routine_Output/HantelTest"
# csvPath = '/Users/v.jayaweera/Documents/SRAvg-ContourDiv-NoInvert.csv'
# acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
# dirPictures = os.listdir(sourcePath)
# imageID = []
# scale = 6.249
# averageSR = []

# if(len(dirPictures)  <= 0):
#     print('The specified folder is empty!')
#     sys.exit()
# else:
    
#     for path in dirPictures:
#         if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):            
#             # Extract contour
#             img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)#
            
#             img = np.array(img)
#             print(img.shape)
            
#             # Create a bitmap from the array
#             bmp = potrace.Bitmap(img)
            
#             # Trace the bitmap to a path
#             path = bmp.trace()
#             # Iterate over path curves
#             x = []
#             y = []
            
#             for curve in path:
#                 print ("start_point =", curve.start_point)
#                 for segment in curve:
#                     print(segment)
#                     end_point_x, end_point_y = segment.end_point.x, segment.end_point.y
#                     if segment.is_corner:
#                         c_x, c_y = segment.c.x, segment.c.y
#                         # x.append(c_x)
#                         # y.append(c_y)
#                         # plt.plot(c_x, c_y, 'r.')
#                     else:
#                         c1_x, c1_y = segment.c1.x, segment.c1.y
#                         c2_x, c2_y = segment.c2.x, segment.c2.y
#                         x.append(c1_x)
#                         y.append(c1_y)
#                         x.append(c2_x)
#                         y.append(c2_y)
#                         # plt.plot(c2_x, c2_y, 'r.')
                        
#                     # plt.plot(x,y, 'b.-')
#                     # plt.plot()
#                     # plt.show()
        
#             k = np.stack((x,y), axis=-1)
#             _, idx = np.unique(k, axis=0,  return_index=True)
#             print(idx.shape)
#             k = k[np.sort(idx)]
            
#             plt.plot(k[:,0], k[:,1], 'g.-')
#             plt.show()