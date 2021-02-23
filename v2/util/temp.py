import xdibias
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score,recall_score,precision_recall_curve
import torch
import pdb
import matplotlib.pyplot as plt


roi_row1 = 20400
roi_row2 = 21400
roi_col1 = 1800
roi_col2 = 2800



test_img = xdibias.Image('/home/wang_yi/la/attn_edge/test_output/attn_edge/result_edges')
bbox = test_img.boundingBox()

coord = np.array([[bbox.left+1800/2,bbox.top-20400/2]])
pix = test_img.map2pix(coord)

imgcrop = xdibias.Image('/home/wang_yi/la/attn_edge/test_output/attn_edge/result_edges', xs=int(pix[0][0]),ys=int(pix[0][1]),width=1000,height=1000)
LOD2_roi = imgcrop.readImageData()
#xdibias.imwrite(LOD2_roi,'/home/wang_yi/la/temp/edge_roi',imgcrop)

'''
data1 = LOD2_roi
data2 = np.zeros_like(data1)

data2[:,:,2] = 0.0
data2[data1[:,:,1]>=0.15,1]=1.0
data2[data1[:,:,1]<0.15,1]=0.0
data2[data1[:,:,0]>=0.05]=(1.0,0.0,0.0)
xdibias.imwrite(data2,'/home/wang_yi/la/temp/edge_ths_roi',imgcrop)
'''

'''
vec_edge = Image.open('/home/wang_yi/la/vectorize/sample2/sample_lines.png')
data_vec = np.asarray(vec_edge)
data3 = np.zeros_like(LOD2_roi)
#pdb.set_trace()
data3[data_vec>127] = 1.0
data3[data_vec<127] = 0.0
xdibias.imwrite(data3,'/home/wang_yi/la/temp/edge_vec_roi',imgcrop)
'''
#pdb.set_trace()








'''
ortho_img = xdibias.Image('/home/wang_yi/la/dataset/WV-1/O.3') #

coord = np.array([[bbox.left+2000/2,bbox.top-20000/2]])
pix = ortho_img.map2pix(coord)

imgcrop = xdibias.Image('/home/wang_yi/la/dataset/WV-1/O.3', xs=int(pix[0][0]),ys=int(pix[0][1]),width=1000,height=1000)
LOD2_roi = imgcrop.readImageData()
xdibias.imwrite(LOD2_roi,'/home/wang_yi/la/temp/ortho_roi',imgcrop)
'''



#dsm_img = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/cgan_dsm_sl1') # > unet, > cgan_urnet
#dsm_img = xdibias.Image('/home/wang_yi/la/unet_dsm/result_unet_dsm') # < cgan_unet
#dsm_img = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/cgan_dsm_w10') # < cgan_unet
#dsm_img = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/result_dsm_w0_fine') # 
#dsm_img = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/result_dsm') #
#dsm_img = xdibias.Image('/home/wang_yi/la/dataset/WV-1/Dfilled_geoid_roi') #

dsm_img = xdibias.Image('/home/wang_yi/la/dataset/LOD2filled_wth_DGM_roi_intrp_GEOKOLref') #

#dsm_img = xdibias.Image('/home/wang_yi/la/temp/version1/result_Coupled_UResNet50_ksenia') #
#dsm_img = xdibias.Image('/home/wang_yi/la/attn_dsm/test_output/attn_dsm/result_dsm') #

coord = np.array([[bbox.left+1800/2,bbox.top-20400/2]])
pix = dsm_img.map2pix(coord)

imgcrop = xdibias.Image('/home/wang_yi/la/dataset/LOD2filled_wth_DGM_roi_intrp_GEOKOLref', xs=int(pix[0][0]),ys=int(pix[0][1]),width=1000,height=1000)
LOD2_roi = imgcrop.readImageData()
#xdibias.imwrite(LOD2_roi,'/home/wang_yi/la/temp/dsm_out_roi',imgcrop)



'''
ndsm_img = xdibias.Image('/home/wang_yi/la/temp/nDSM_out_roi')
data_ndsm = ndsm_img.readImageData()
xdibias.imwrite(data_ndsm,'/home/wang_yi/la/temp/nDSM_out_roi_geo',imgcrop)
'''


gt_ndsm_img = xdibias.Image('/home/wang_yi/la/dataset/nLOD2filled_wth_DGM_roi_intrp_GEOKOLref')
coord = np.array([[bbox.left+1800/2,bbox.top-20400/2]])
pix = gt_ndsm_img.map2pix(coord)
imgcrop = xdibias.Image('/home/wang_yi/la/dataset/nLOD2filled_wth_DGM_roi_intrp_GEOKOLref', xs=int(pix[0][0]),ys=int(pix[0][1]),width=1000,height=1000)
data_gt_ndsm_roi = imgcrop.readImageData()
#xdibias.imwrite(data_gt_ndsm_roi,'/home/wang_yi/la/temp/nDSM_gt_roi_geo',imgcrop)

data_out_ndsm_roi = LOD2_roi - data_gt_ndsm_roi
xdibias.imwrite(data_out_ndsm_roi,'/home/wang_yi/la/temp/DTM_roi_geo',imgcrop)



'''
out_mask = Image.open('/home/wang_yi/la/vectorize/sample2/sample_polygons_out.png')
data_out_mask = np.asarray(out_mask)
data3 = np.zeros_like(data_out_mask[:,:,0],dtype='float32')
#pdb.set_trace()
data3[data_out_mask[:,:,0]>100] = 1.0
data3[data_out_mask[:,:,0]<100] = 0.0
xdibias.imwrite(data3,'/home/wang_yi/la/temp/mask_out_roi',imgcrop)
'''


'''
gt_mask = xdibias.Image('/home/wang_yi/la/dataset/BMdirectFromLOD2filled_roi_GEOKOL')
coord = np.array([[bbox.left+1800/2,bbox.top-20400/2]])
pix = gt_mask.map2pix(coord)
imgcrop = xdibias.Image('/home/wang_yi/la/dataset/BMdirectFromLOD2filled_roi_GEOKOL', xs=int(pix[0][0]),ys=int(pix[0][1]),width=1000,height=1000)
data_gt_mask = imgcrop.readImageData()
#pdb.set_trace()
data_gt_mask_roi = np.zeros_like(data_gt_mask,dtype='float32')
data_gt_mask_roi[data_gt_mask>127] = 1.0
data_gt_mask_roi[data_gt_mask<127] = 0.0
xdibias.imwrite(data_gt_mask_roi,'/home/wang_yi/la/temp/mask_gt_roi',imgcrop)

img_gt_mask = Image.fromarray((data_gt_mask_roi*255.0).astype('uint8'))
img_gt_mask.save('mask_gt.png')

#pdb.set_trace()

'''








#xx = xdibias.Image('/home/wang_yi/la/temp/ortho_roi')
#yy = xx.readImageData()
#pdb.set_trace()

