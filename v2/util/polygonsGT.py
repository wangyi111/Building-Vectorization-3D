# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry.polygon import LinearRing
from shapely import geometry
from scipy import spatial

import cv2
import pdb

import os

import xdibias

import skimage.transform

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

import scipy as sp
import scipy.ndimage
from scipy.spatial import distance as dist

import glob 

def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    pdb.set_trace()
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros
    pdb.set_trace()
    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array
    
# import the necessary packages
def isInside(point_coord, size_1, size_2):
    return 0 <= point_coord[0] <= size_1 and\
           0 <= point_coord[1] <= size_2
 


def read_object_instances(input_file):
    empty_lines = 0
    blocks = []
    block = []
    object_instance = []

    with open(input_file) as f:
        lines = f.readlines()  
        #pdb.set_trace()
        #for line in open(input_file):
        for line in lines[:-1]:
        #for line in lines[710974:-1]:
        #for line in lines[1497194:-1]:
        
            # Check for new instance. It is defined by #
            if line.startswith('#'):
                #pdb.set_trace()
                if blocks:
                    object_instance.append(blocks)
                    blocks = []
                continue
            # Check for empty/commented lines
            if line.isspace():
                #pdb.set_trace()
                # If 1st one: new block
                if empty_lines == 0:
                    #pdb.set_trace()
                    if block:
                        blocks.append(block)
                        block = []
                empty_lines += 1
            # Non empty line: add line in current(last) block
            elif line.startswith('R'):
                empty_lines = 0
                row =  line.split(" ")
                
                points = [float(row[1]), float(row[2]), float(row[3])] 
                block.append(points)
    #print len(blocks)
    return object_instance
    
    
list_of_classes = []    
## read from one file 
#poly = read_object_instances('/home/davy_ks/la/3d-gm-lod2_Berlin/Charlottenburg-Wilmersdorf/RoofSurf.dat')
#pdb.set_trace()

rootdir = '/home/wang_yi/la/dataset/Berlin_LoD2'
buildings = []
#poly=[]


obj_id = 1

for subdir, dirs, files in os.walk(rootdir):
    
    for one_dir in dirs:
        
        path = os.path.join(subdir,one_dir)      

        for sub_folder, sub_dirs, sub_files in os.walk(path):
            
            for file in sub_files:
                #if sub_folder != '/home/wang_yi/la/dataset/Berlin_LoD2/Friedrichshain-Kreuzberg': # exist plygon with only 2 corners
                    #pass
                if file == 'RoofSurf.dat':
    
                    print sub_folder          
                    #pdb.set_trace()
                    
                    
                    
                    match = glob.glob("/home/davy_ks/la/Berlin/LOD2_allBerlin/LOD2DSM_{0}*".format(os.path.basename(path)[:4]))
                    print match
                    filesplit = os.path.basename(match[0]).split(".")                            
                            
                    temp = xdibias.Image("/home/davy_ks/la/Berlin/LOD2_allBerlin/{0}".format(filesplit[0]))

                    IMG = xdibias.Image('/home/wang_yi/la/MakeGT/polygon_instances/polygon_instances_{0}'.format(os.path.basename(path)), createNew = True)

                    IMG.Rows = temp.Rows
                    IMG.Columns = temp.Columns
                    IMG.Channels = temp.Channels
                    IMG.BitsPerChannel = 32
                    IMG.copyMetadata(temp,
                                     imagetype = True,
                                     geo = True,
                                     radiometry = True)                          
                     
                    #pdb.set_trace()
                    img_in = np.zeros((temp.Rows,temp.Columns), dtype = np.float32)
                    #poly.append(read_blocks(os.path.join(subdir,file),list_of_classes))
                    poly = read_object_instances(os.path.join(path,file))
                    
                    param = 0
                    
                    for el in range(0,len(poly)):
                    #for el in range(0,1000):
                        #pdb.set_trace()            
                        if len(poly[el]) == 0:
                            pdb.set_trace()
                            pass
                        
    #                    elif poly[el][0][1] == 0 or poly[el][0][1]==9999:
    #                        pass
                            
                        #else:
                        else:
                            
                            for sub_el in range(len(poly[el])):
    
                                # Create array of [x,y]             
                                points = np.array([(poly[el][sub_el][j][0],poly[el][sub_el][j][1]) for j in range(0,len(poly[el][sub_el]))])      
                                
                                unique_index = list(range(len(points)))
                                
                                ind_duplicates = []
    
                                for ind in range(0,len(points[:-1])): 
                                    arr  = spatial.KDTree(points[ind+1:]).query_ball_point(points[ind], 1e-7)                                                      
                                    if not arr:
                                        pass
                                    else:
                                        if not ind_duplicates:
                                            ind_duplicates = [d + (ind+1) for d in arr]
                                        else:
                                            if ind_duplicates.count(ind) == 1:
                                                pass
                                            else:
                                                for index in arr: ind_duplicates.append(index+(ind+1))
                                                    
                                sorted(ind_duplicates, key=int)
                                                    
                                sorted_points = [p for ind, p in enumerate(points) if ind not in ind_duplicates]
                                pix_points=np.array([[[int((pnt[0]-temp.XGeoRef)/0.5),int((temp.YGeoRef-pnt[1])/0.5)]] for pnt in sorted_points])
    
                                #pdb.set_trace()
                                x = np.array([round(float(pp[0][0])) for pp in pix_points])
                                y = np.array([round(float(pp[0][1])) for pp in pix_points])
                                x = x.astype('int')
                                y = y.astype('int')        
                                rr_in, cc_in = skimage.draw.polygon(y, x) # exclude edges
                                                                                                
                                img_in[rr_in, cc_in] = obj_id # within polygon
                                obj_id += 1
                                
                            
                    #pdb.set_trace() 
                    IMG.Background = 0
                    IMG.writeImage(img_in)

                    


                
                
print "last id", obj_id
#print "END"                
#print list_of_classes.sort()
#pdb.set_trace()
