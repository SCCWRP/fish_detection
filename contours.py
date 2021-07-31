import numpy as np
import cv2 as cv # opencv 
#from pandas import Series, DataFrame, isnull, concat
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from numpy import random, nanmax, nanmin, argmax, unravel_index
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import distance_transform_edt
# We'll have to use tensorflow rather than imageai to do the object detection in the contour class
#import imageai
#from imageai.Detection.Custom import CustomObjectDetection # github.com/olafenwamoses/imageai
import time
import os
import sys
import re
import argparse
import copy
import glob

class Contour:
    def __init__(self, points, original_image):
        "Original image should be the numpy array version, not the path to the image"
        self.points = points
        self.original_image = original_image
        self.cm_pixel_ratio = 1 # default. can be set later
    def crop_window(self, path, image, cushion=0):
        self.max_x = nanmax([x[0] + cushion for x in self.points] + [image.shape[1]])
        self.max_y = nanmax([y[1] + cushion for y in self.points] + [image.shape[0]])
        self.min_x = nanmin([x[0] - cushion for x in self.points] + [0])
        self.min_y = nanmin([y[1] - cushion for y in self.points] + [0])
        self.window = image[self.min_x:self.max_x, self.min_y:self.max_y]
        #cv.imwrite(path, self.window)
        cv.imwrite(path, image)
        return None
    def containsOysters(self, path, detector):
        pass
    def matchOysterShape(self, contour_to_match, max_score = 0.3):
        match = cv.matchShapes(contour_to_match, self.points, 1, 0.0)
        print("match score: %s" % match)
        if match > max_score:
            print("the shape of this contour does not sufficiently match the shape of an oyster")
            return False
        else:
            return True
    def getLength(self):
        '''
        Using the method of getting the max distance across and a nearly orthogonal vector of max distance to that one
        Hard to explain in words
        '''
        # we want to loop through the points contained in p for each contour. 
        # p is a set of points that form the contour so the contours contains a set of p per contour drawn
        # compute distance
        D = pdist(self.points)
        # input into an array
        D = squareform(D);
        # find max distance and which points this corresponds to
        self.pixellength = round(nanmax(D), 2)
        # called I_row and I_col since it is grabbing the "ij" location in a matrix where the max occurred.
        # the row number where it occurs represents one of the indices in the original self.points array where one of the points on the contour lies
        # the column number would be the point on the opposite side of the contour
        # L_row, and L_col since these indices correspond with coordinate points that give us the length
        [L_row, L_col] = unravel_index( argmax(D), D.shape )
        
        self.min_length_coord = tuple(self.points[L_row])
        self.max_length_coord = tuple(self.points[L_col])
        self.length_coords = [self.min_length_coord, self.max_length_coord]
        self.length_vector = np.array(self.max_length_coord) - np.array(self.min_length_coord)
        self.unit_length_vector = self.length_vector / norm(self.length_vector)      
        self.length = round(self.pixellength * self.cm_pixel_ratio, 2) # px * mm / px yields units of mm
        return self.length
    def getWidth(self):
        # length axis represents a unit vector along the direction where we found the longest distance over the contour
        # length_axis = (np.array(p[L_col]) - np.array(p[L_row])) / norm(np.array(p[L_col]) - np.array(p[L_row]))
        '''above will be replaced with self.unit_length_vector'''
        # length_axis = self.unit_length_vector
       
        # all_vecs will be an list of vectors that are all the combinations of vectors that pass over the contour area
        all_vecs = []
        coordinates = []
        for i in range(0, len(self.points) - 1):
            for j in range(i + 1, len(self.points)):
                all_vecs.append(np.array(self.points[i]) - np.array(self.points[j]))
                coordinates.append([tuple(self.points[i]), tuple(self.points[j])])
        
        # make it a column of a pandas dataframe
        #vectors_df = DataFrame({'all_vecs': all_vecs, 'coordinates': coordinates})
        vectors_df = {'all_vecs': all_vecs, 'coordinates': coordinates}
        
        # Here we normalize all those vectors to prepare to take the dot product with the vector called "length vector"
        # Dot product will be used to determine orthogonality
        vectors_df['all_vecs_normalized'] = vectors_df.all_vecs.apply(lambda x: x / norm(x))

        # Take the dot product
        #vectors_df['dot_product'] = vectors_df.all_vecs_normalized.apply(lambda x: np.dot(x, length_axis))
        vectors_df['dot_product'] = vectors_df.all_vecs_normalized.apply(lambda x: abs(np.dot(x, self.unit_length_vector)))
        #vectors_df['orthogonal'] = vectors_df.dot_product.apply(lambda x: x < 0.15)
       
        vectors_df['norms'] = vectors_df.all_vecs.apply(lambda x: norm(x))

        if any(vectors_df.dot_product < 0.075):
            # allowing dot product to be up to 0.15 allows the length and width to have an angle of 81.37 to 90 degrees between each other
            width = nanmax(vectors_df[vectors_df.dot_product < 0.075].norms)
            self.width_coords = vectors_df[vectors_df.norms == width].sort_values('dot_product').coordinates.tolist()[0]
            self.pixelwidth = round(width, 2)
            self.width = round(self.pixelwidth * self.cm_pixel_ratio, 2) # pixels times cm / pixels yields units of millimeters
        else:
            self.pixelwidth = None
            self.width_coords = None
    
    def getArea(self):
        self.surfacearea_px2 = cv.contourArea(self.points)
        self.surfacearea = cv.contourArea(self.points) * (self.cm_pixel_ratio ** 2)

    def drawLength(self, image):
        "image represents the image we are drawing on"
        cv.line(image, self.length_coords[0], self.length_coords[1], (0,255,0))
        #cv.line(image, self.width_coords[0], self.width_coords[1], (0,255,0))
        #cv.putText(image, "L:%scm, W:%scm" % (self.length, self.width), self.length_coords[1], cv.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        return None
