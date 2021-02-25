'''
vectorize raster edge/corner
input: raster edge [0,255,0] / corner [255,0,0]
output: list of edges (corner1[pr,pc], corner2[pr,pc])         


'''

import numpy as np
import xdibias
import skimage
import pdb
from PIL import Image
from tqdm import tqdm
import math
import shapely
import shapely.ops
import torch
from skimage import measure


class Corner():
    def __init__(self,pr,pc):
        self.pr = pr # row
        self.pc = pc # col
        self.geo_x = None
        self.geo_y = None
        self.value = 1
        self.height = None
        self.ID = None

    def set_geo(self,geo_x,geo_y):
        self.geo_x = geo_x
        self.geo_y = geo_y

    def set_height(self,height):
        self.height = height


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
    #rrl,ccl = line_thick(4,c1,c2,shape)
    rr,cc = line_thick(5,c1,c2,shape)
    buffer = np.zeros_like(img[:,:,0])
    buffer[rr,cc] = 1
    arr = edge_bin * buffer
    edge_labels = measure.label(arr)
    #pdb.set_trace()
    if c1.ID != c2.ID:
        return False
    elif np.count_nonzero(img[rr,cc,0]>=0.05)>2:
        #pdb.set_trace()
        return False
    #elif np.mean(img[rrl,ccl,1])<0.5:
    
    elif (edge_labels[c1.pr,c1.pc] != edge_labels[c2.pr,c2.pc]) & (np.mean(img[rrl,ccl,1])<0.5):
        return False
    else:
        #pdb.set_trace()
        return np.mean(img[rrl,ccl,1])
    
def xy2rc(x,y):
    return -y,x


def xy2geo(x,y):
    r = -y
    c = x
    left_x = bbox.left
    up_y = bbox.top

    geo_x = left_x + c*1.0
    geo_y = up_y - r*1.0
    return geo_x,geo_y



''' load input raster image '''

roi_row1 = 0
roi_row2 = 1007
roi_col1 = 0
roi_col2 = 1182

### edge
#test_img = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/unet_only/result_edges')
#test_img = xdibias.Image('/home/wang_yi/la/UNet_edge2/test_output/dsm_ks/result_edges')
test_img = xdibias.Image('/home/wang_yi/la/attn_edge/test_output/attn_edge2/result_edges')
bbox = test_img.boundingBox()

data = test_img.readImageData()

data1 = data[roi_row1:roi_row2,roi_col1:roi_col2,:]

#data1 = torch.nn.Softmax(dim=-1)(torch.tensor(data1)).data.numpy()
data2 = data1.copy()

data2[:,:,2] = 0.0
data2[data1[:,:,1]>=0.15,1]=1.0
data2[data1[:,:,1]<0.15,1]=0.0
#data2[data1[:,:,0]>=0.05]=(1.0,0.0,0.0)
data2[data1[:,:,0]<0.05,0]=0.0


### building ID
edge_bin = np.zeros_like(data2[:,:,0])
edge_bin[data2[:,:,0]>0]=1
edge_bin[data2[:,:,1]>0]=1
building_labels = measure.label(edge_bin)


### dsm
#kdsm_img = xdibias.Image('/home/wang_yi/la/temp/version1/result_Coupled_UResNet50_ksenia')
kdsm_img = xdibias.Image('/home/wang_yi/la/attn_dsm/test_output/attn_dsm2/result_dsm')
xdibias.geo.intersectRect(bbox,kdsm_img.boundingBox())
kdsm_roi = kdsm_img.getROIImage(bbox,gridtol=0.5)

data_kdsm = kdsm_roi.readImageData()
data9 = data_kdsm[roi_row1:roi_row2,roi_col1:roi_col2]





''' extract corners '''
## find all corners
corners_raw = []
for row in range(data2.shape[0]):
    for col in range(data2.shape[1]):
        if data2[row,col,0]>=0.05:
            corner_i = Corner(row,col)
            corner_i.value = data2[row,col,0]
            corner_i.ID = building_labels[row,col]
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


#pdb.set_trace()
data3[data3[:,:,0]>=0.05] = (1.0,0.0,0.0)
img = Image.fromarray((data3*255.0).astype('uint8'))
img.save('corner_nms7.png')

#pdb.set_trace()

## set height
for i,c1 in enumerate(corners):
    #pdb.set_trace()
    window = data9[(c1.pr-2):(c1.pr+2),(c1.pc-2):(c1.pc+2)]
    if window.shape == (4,4):
        height = window.max()
    else:
        height = data9[c1.pr,c1.pc]
    c1.set_height(height)





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
img.save('sample_lines.png')



''' polygons '''
#pdb.set_trace()
out2 = np.zeros(data2.shape)

lines_xy = []
for i,edge in enumerate(edges):
    
    line_xy = ( (edge.left.pc,-edge.left.pr), (edge.right.pc,-edge.right.pr) )
    lines_xy.append(line_xy)

#pdb.set_trace()
polygons_xy = list(shapely.ops.polygonize(lines_xy))

polygons_outline = list(shapely.ops.unary_union([p for p in polygons_xy if p.is_valid]))

polygons = []
for i,polygon_outline in enumerate(polygons_outline):
    polygon = list((shapely.ops.transform(xy2rc,polygon_outline)).exterior.coords)    
    polygons.append(polygon)

    rc = np.asarray(polygon)
    r = rc[:,0]
    c = rc[:,1]
    rr,cc = skimage.draw.polygon(r,c)
    out2[rr,cc] = (0.5,0.5,0.5)

#pdb.set_trace()
img2 = Image.fromarray((out2*255).astype('uint8'))
img2.save('sample_polygons_out.png')

#pdb.set_trace()
import fiona
from shapely.geometry import mapping

schema = {'geometry':'Polygon','properties':{'id':'int'}}

with fiona.open('sample_polys_in.shp','w','ESRI Shapefile',schema) as c:
    for i,poly in enumerate(polygons_xy):
        c.write({'geometry':mapping(shapely.ops.transform(xy2rc,poly)),'properties':{'id':i}})

with fiona.open('sample_polys_out.shp','w','ESRI Shapefile',schema) as c:
    for i,poly in enumerate(polygons_outline):
        c.write({'geometry':mapping(shapely.ops.transform(xy2rc,poly)),'properties':{'id':i}})

#pdb.set_trace()
with fiona.open('sample_polys_in_geo.shp','w','ESRI Shapefile',schema) as c:
    for i,poly in enumerate(polygons_xy):
        c.write({'geometry':mapping(shapely.ops.transform(xy2geo,poly)),'properties':{'id':i}})

with fiona.open('sample_polys_out_geo.shp','w','ESRI Shapefile',schema) as c:
    for i,poly in enumerate(polygons_outline):
        c.write({'geometry':mapping(shapely.ops.transform(xy2geo,poly)),'properties':{'id':i}})    


    
    
