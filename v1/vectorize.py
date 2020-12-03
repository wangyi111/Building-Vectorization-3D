'''
vectorize raster edge/corner
input: raster edge [0,255,0] / corner [255,0,0]
         


'''

import numpy as np
import xdibias
import skimage
import pdb
from PIL import Image
from tqdm import tqdm
import math

class Corner():
    def __init__(self,pr,pc):
        self.pr = pr # row
        self.pc = pc # col
        self.geo_x = None
        self.geo_y = None
        self.value = 1

    def set_geo(self,geo_x,geo_y):
        self.geo_x = geo_x
        self.geo_y = geo_y


class Edge():
    def __init__(self,left,right):
        self.left = left
        self.right = right
        self.value = 1

def line_thick(thick,c1,c2,shape):
    x1 = c1.pc
    y1 = c1.pr
    x2 = c2.pc
    y2 = c2.pr
    if x2==x1:
        a = math.pi/2
    else:
        a = math.atan((y2-y1)/(x2-x1))
    sin = math.sin(a)
    cos = math.cos(a)
    xdelta = sin * thick / 2.0
    ydelta = cos * thick / 2.0
    xx1 = x1-xdelta
    yy1 = y1+ydelta
    xx2 = x1+xdelta
    yy2 = y1-ydelta
    xx3 = x2+xdelta
    yy3 = y2-ydelta
    xx4 = x2-xdelta
    yy4 = y2+ydelta
    r = np.array([yy1,yy2,yy3,yy4],dtype='int64')
    c = np.array([xx1,xx2,xx3,xx4],dtype='int64')
    rr1,cc1 = skimage.draw.polygon(r,c,shape=shape)
    rr2,cc2 = skimage.draw.polygon_perimeter(r,c,shape=shape)
    
    rr3 = np.concatenate((rr1,rr2))
    cc3 = np.concatenate((cc1,cc2))
    rc1 = np.array([rr3,cc3])
    rc2 = np.unique(rc1,axis=-1)
    rr = rc2[0]
    cc = rc2[1]
    
    return rr,cc

def IsEdge(c1,c2,img):
    rrl,ccl,val = skimage.draw.line_aa(c1.pr,c1.pc,c2.pr,c2.pc) # row,col
    shape = img.shape[0:2]
    rr,cc = line_thick(5,c1,c2,shape)
    #pdb.set_trace()
    if np.count_nonzero(img[rr,cc,0]>=0.05)>2:
        #pdb.set_trace()
        return False
    elif np.mean(img[rrl,ccl,1])<0.7:
        return False
    else:
        #pdb.set_trace()
        return np.mean(img[rrl,ccl,1])
    





''' load input raster image '''

roi_row1 = 40000
roi_row2 = 41000
roi_col1 = 2000
roi_col2 = 3000

test_img = xdibias.Image('/home/wang_yi/la/UNet_edge2/test_output/result_edges')
data = test_img.readImageData()

data1 = data[roi_row1:roi_row2,roi_col1:roi_col2,:]
data2 = data1.copy()

data2[:,:,2] = 0.0
data2[data1[:,:,1]>=0.15,1]=1.0
data2[data1[:,:,1]<0.15,1]=0.0
#data2[data1[:,:,0]>=0.05]=(1.0,0.0,0.0)
data2[data1[:,:,0]<0.05,0]=0.0



''' extract corners '''
## find all corners
corners_raw = []
for row in range(data2.shape[0]):
    for col in range(data2.shape[1]):
        if data2[row,col,0]>=0.05:
            corner_i = Corner(row,col)
            corner_i.value = data2[row,col,0]
            corners_raw.append(corner_i)
print 'raw corners number: ', len(corners_raw)
#pdb.set_trace()

## non-maximum supression
data3 = data2.copy()
for i,c1 in enumerate(corners_raw):
    for j,c2 in enumerate(corners_raw):
        if c2 == c1:
            continue
        if c2.pr<c1.pr+3 and c2.pr>c1.pr-3 and c2.pc<c1.pc+3 and c2.pc>c1.pc-3:
            if c2.value < c1.value:
                c2.value = 0
                data3[c2.pr,c2.pc,0] = 0
data3[data3[:,:,0]>=0.05] = (1.0,0.0,0.0)
corners = []
for i,c1 in enumerate(corners_raw):
    if c1.value == 0:
        pass
    else:
        corners.append(c1)
        
print 'filtered corners number: ', len(corners)

'''
pdb.set_trace()
data3[data3[:,:,0]>=0.05] = (1.0,0.0,0.0)
img = Image.fromarray((data3*255.0).astype('uint8'))
img.save('corner_nms7.png')
'''


''' build graph/find edges '''
num_c = len(corners)
#N = np.zeros([num_c,num_c])
edges = []

for i in tqdm(range(num_c)):
    c1 = corners[i]
    for j in tqdm(range(i,num_c)):        
        c2 = corners[j]
        #pdb.set_trace()
        value = IsEdge(c1,c2,data3)
        if value:
            #pdb.set_trace()
            #N[i,j] = 1
            edge = Edge(c1,c2)
            edge.value = value
            edges.append(edge)

print len(edges)



#pdb.set_trace()
''' draw edges '''
out = np.zeros(data2.shape)

for i,edge in enumerate(edges):
    rr,cc = skimage.draw.line(edge.left.pr,edge.left.pc,edge.right.pr,edge.right.pc)
    out[rr,cc] = (0,1,0)
    out[edge.left.pr,edge.left.pc] = (1,0,0)
    out[edge.right.pr,edge.right.pc] = (1,0,0)
    

img = Image.fromarray((out*255).astype('uint8'))
img.save('output/output_visual.png')



    
    
