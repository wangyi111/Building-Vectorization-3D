import xdibias
import pdb
import pylab as pl
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.mlab import griddata
from matplotlib.patches import Circle
import xdibias
import cv2
import scipy.ndimage as ndimage
import os
import glob
import scipy as sp


class getCol:
    matrix = []
    def __init__(self, file, delim=" "):
        with open(file, 'rU') as f:
            getCol.matrix =  [filter(None, l.split(delim)) for l in f]

    def __getitem__ (self, key):
        #print "key",key
        column = []
        for row in getCol.matrix:
            if row[0]!=("#") and row[0]!= '\n':
                try:
                    column.append(row[key])
                except IndexError:
                    # pass
                    column.append("")
        return column


def isInside(point_coord, size_1, size_2):
    return 0 <= point_coord[0][1] <= size_1 and\
           0 <= point_coord[0][0] <= size_2
           
           
rootdir = '/home/wang_yi/la/dataset/Berlin_LoD2' # dir of .dat files (to make GT corner map)
x = []
y = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file == 'RoofSurf.dat':
            #pdb.set_trace()
            print os.path.join(subdir,file)
            x += map(float,getCol(os.path.join(subdir,file))[1]) # list: corner points x-corrd.
            y += map(float,getCol(os.path.join(subdir,file))[2]) # list: corner points y-coord.
            


img = xdibias.Image("/home/wang_yi/la/dataset/WV-1/Dfilled_geoid_roi") # stereo dsm image
Dfilled = img.readImageData()
print "max(Dfilled) = ", np.max(Dfilled), "min(Dfilled) = ", np.min(Dfilled) # height range


# generate ground truth (gt) image
# where white pixel represent one building corner
gt_image = np.zeros((Dfilled.shape), dtype = np.float32)
print gt_image.shape
print gt_image.dtype


for xx,yy in zip(x,y):
    coord = np.array([[xx,yy]])
    pix = img.map2pix(coord) # transform map coord. to pixel coord. based on reference image's meta information
    #pdb.set_trace()
    if isInside(pix, Dfilled.shape[0],Dfilled.shape[1]):
        pix_index = np.floor(pix).astype('int32')
        #pdb.set_trace()
        gt_image[pix_index[0][1],pix_index[0][0]] = 255.0 # valuate all corners as white


#plt.figure(1)
#plt.imshow(gt_image[7000:8000, 13000:14000],cmap='gray')


gt_map = np.zeros((Dfilled.shape), dtype = np.float32)
gt_map[gt_image==255.0]=1.0

# why implement gaussian?
#sigma = 3
#gt_map = ndimage.filters.gaussian_filter(gt_map, sigma)
#gt_map -= np.min(gt_map)
#gt_map /= np.max(gt_map)

#print gt_map.max(), gt_map.min()

#pdb.set_trace()
#plt.figure(2)
#plt.imshow(gt_map[7000:8000, 13000:14000],cmap='gray')
#plt.show()

outputdir = '/home/wang_yi/la/dataset/gt_corepoints'
# save samples to jpg
#sp.misc.imsave(outputdir + "/gt_corner_img.jpg", gt_image[7000:8000, 13000:14000])
#sp.misc.imsave(outputdir + "/gt_corner_map.jpg", gt_map[7000:8000, 13000:14000])

# save whole image to png
sp.misc.imsave(outputdir + "/gt_corner_img_all.png", gt_image)
sp.misc.imsave(outputdir + "/gt_corner_map_all.png", gt_map)

#save whole image to xdibias
#imgout = xdibias.Image(outputdir + "/gt_corner_img" ,createNew=True) 
#xdibias.imwrite(gt_image, outputdir + "/gt_corner_img", img)

#imgout = xdibias.Image(outputdir + "/gt_corner_map" ,createNew=True) 
#xdibias.imwrite(gt_map, outputdir + "/gt_corner_map", img)


print "END"       
#pdb.set_trace()


#x = map(float,getCol("/home/davy_ks/la/3d-gm-lod2_Berlin/Charlottenburg-Wilmersdorf/RoofSurf.dat")[1])
#y = map(float,getCol("/home/davy_ks/la/3d-gm-lod2_Berlin/Charlottenburg-Wilmersdorf/RoofSurf.dat")[2])
#z = map(float,getCol("/home/davy_ks/la/Charlottenburg-Wilmersdorf/1build.dat")[3])

#
#fig= plt.figure()
#
#ax= fig.add_subplot(111, projection ='3d')
#x,y = np.meshgrid(x,y)
#ax.plot_surface(x, y, z)

#mpl.rcParams['legend.fontsize'] = 10

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(x, y, z, label='parametric curve')
#ax.legend()

#ax.scatter(x, y, z, c='r', marker='o')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#plt.show()

#plt.plotfile('/home/davy_ks/la/Charlottenburg-Wilmersdorf/1build.dat', delimiter=' ', cols=(1, 2, 3), names=('col1', 'col2','col3'), marker='o')
#plt.show()
     
#with open('/home/davy_ks/la/Charlottenburg-Wilmersdorf/1build.dat') as f:
#    lines = [line for line in f if line.strip()]
#
#data = [line.split() for line in lines]
#plt.plot(*zip(*data), marker='o', color='r', ls='')
#plt.show()               
 
#def read_blocks(input_file):
#    empty_lines = 0
#    blocks = []
#    for line in open(input_file):
#        # Check for empty/commented lines
#        if line.isspace() or line.startswith('#'):
#            # If 1st one: new block
#            #pdb.set_trace()
#            if empty_lines == 0:
#                blocks.append([])
#            empty_lines += 1
#        # Non empty line: add line in current(last) block
#        else:
#            empty_lines = 0
#            blocks[-1].append(line)
#    return blocks
#
##for block in read_blocks('/home/davy_ks/la/Charlottenburg-Wilmersdorf/1build.dat'):
##    print '-> block'
##    for line in block:
##        print line
#
#
#with open('/home/davy_ks/la/Charlottenburg-Wilmersdorf/1build.dat') as f:
#    lines = f.read()
#
#blocks = lines.split('\n\n')
#for block in blocks:
#    pdb.set_trace()
#    data = [line.split() for line in block.splitlines()]
#    plt.plot(*zip(*data), marker='o', color='r', ls='')
#    plt.show()



#fig,ax = plt.subplots(1)
#ax.set_aspect('equal')
#ax.imshow(Dfilled[orig_y-500:orig_y+500, orig_x-500:orig_x+500], cmap='gray')
##ax.imshow(Dfilled, cmap='gray')

#for xx,yy in zip(x,y):
#    #pdb.set_trace()
#    #print xx, yy
#    coord = np.array([[xx,yy]])
#    pix = img.map2pix(coord)
##    r_xx = int(round(pix[0][0], 0))
##    r_yy = int(round(pix[0][1], 0))
##    print r_xx, r_yy
#    #pdb.set_trace()
#    circ = Circle((pix[0][0]-(orig_x-500),pix[0][1]-(orig_y-500)),1)
#    circ.set_edgecolor('white')
#    circ.set_facecolor('white')
#    #circ = Circle((pix[0][0],pix[0][1]),1)
#    ax.add_patch(circ)

