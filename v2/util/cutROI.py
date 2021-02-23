# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:35:36 2017

@author: davy_ks
"""

import xdibias
import numpy as np 
import pdb
#import gdal
#import matplotlib.pyplot as plt
'''
###############################################################################
#                                                                             #
#                         LOD2 Berlin                                         #
#                                                                             #
###############################################################################
'''

path_to_LOD = '/home/wang_yi/la/MakeGT'
DSM = xdibias.Image('/home/wang_yi/la/MakeGT/DfilledGeokol')

## Define bounding box
bbox = DSM.boundingBox()
print bbox.left,bbox.right, bbox.bottom, bbox.top

#Read image from which we will cut 
#LOD2 = xdibias.Image(path_to_LOD + '/building_instances_all')
#LOD2 = xdibias.Image(path_to_LOD + '/building_edges_all')
#LOD2 = xdibias.Image(path_to_LOD + '/building_edges_2c_all')
LOD2 = xdibias.Image(path_to_LOD + '/building_polygons_all')

coord = np.array([[bbox.left,bbox.top]])
pix = LOD2.map2pix(coord)

#imgcrop = xdibias.Image(path_to_LOD + '/building_instances_all', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#imgcrop = xdibias.Image(path_to_LOD + '/building_edges_all', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#imgcrop = xdibias.Image(path_to_LOD + '/building_edges_2c_all', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
imgcrop = xdibias.Image(path_to_LOD + '/building_polygons_all', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)

LOD2_roi = imgcrop.readImageData()

#xdibias.imwrite(LOD2_roi, path_to_LOD + '/building_instances_roi', imgcrop)
#xdibias.imwrite(LOD2_roi, path_to_LOD + '/building_edges_roi', imgcrop)
#xdibias.imwrite(LOD2_roi, path_to_LOD + '/building_edges_2c_roi', imgcrop)
xdibias.imwrite(LOD2_roi, path_to_LOD + '/building_polygons_roi', imgcrop)


#'''
################################################################################
##                                                                             #
##                         LOD2 Nord part of Munich                            #
##                                                                             #
################################################################################
#'''
#path_to_LOD = '/home/davy_ks/la/data'
#DSM = xdibias.Image(path_to_LOD + '/Geländemodell_roi')
##DSM = xdibias.Image('/home/davy_ks/la/data/LOD2_Munich_Format_XDIBIAS_UTM32/LOD2nordMUC')
##DSM = xdibias.Image(path_to_LOD + '/DGM0.5m_smroi')
#
### Define bounding box
#bbox = DSM.boundingBox()
#print bbox.left,bbox.right, bbox.bottom, bbox.top
#
##Read image from which we will cut 
##LOD2 = xdibias.Image(path_to_LOD + '/LOD2_allBerlin/filledData/LOD2filled')
#LOD2 = xdibias.Image('/home/davy_ks/la/data/LOD2_Munich_Format_XDIBIAS_UTM32/LOD2nordMUC')
##DGM = xdibias.Image(path_to_LOD + '/Dfilled_Berlin_roi')
#
#
#coord = np.array([[bbox.left,bbox.top]])
#pix = LOD2.map2pix(coord)
##pix = DGM.map2pix(coord)
#
#
##imgcrop = xdibias.Image(path_to_LOD + '/LOD2_allBerlin/filledData/LOD2filled_wth_DGM_roi_intrp', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#imgcrop = xdibias.Image('/home/davy_ks/la/data/LOD2_Munich_Format_XDIBIAS_UTM32/LOD2nordMUC', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#LOD2_roi = imgcrop.readImageData()
#xdibias.imwrite(LOD2_roi, '/home/davy_ks/la/data/LOD2_Munich_Format_XDIBIAS_UTM32/LOD2nordMUC_roi', imgcrop)


#'''
################################################################################
##                                                                             #
##                         LOD2 Patch of Munich                                #
##                                                                             #
################################################################################
#'''
#path_to_LOD = '/home/davy_ks/la/pytorch-CycleGAN-and-pix2pix/DSM2DOM/RS_Journal'
#DSM = xdibias.Image(path_to_LOD + '/MUC_Dfilled_Patch2')
##DSM = xdibias.Image('/home/davy_ks/la/data/LOD2_Munich_Format_XDIBIAS_UTM32/LOD2nordMUC')
##DSM = xdibias.Image(path_to_LOD + '/DGM0.5m_smroi')
#
### Define bounding box
#bbox = DSM.boundingBox()
#print bbox.left,bbox.right, bbox.bottom, bbox.top
#
##Read image from which we will cut 
##LOD2 = xdibias.Image(path_to_LOD + '/LOD2_allBerlin/filledData/LOD2filled')
#LOD2 = xdibias.Image('/home/davy_ks/la/data/LOD2_Munich_Format_XDIBIAS_UTM32/LOD2nordMUCwthDGM_roi')
##DGM = xdibias.Image(path_to_LOD + '/Dfilled_Berlin_roi')
#
#
#coord = np.array([[bbox.left,bbox.top]])
#pix = LOD2.map2pix(coord)
##pix = DGM.map2pix(coord)
#
#
##imgcrop = xdibias.Image(path_to_LOD + '/LOD2_allBerlin/filledData/LOD2filled_wth_DGM_roi_intrp', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#imgcrop = xdibias.Image('/home/davy_ks/la/data/LOD2_Munich_Format_XDIBIAS_UTM32/LOD2nordMUCwthDGM_roi', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#LOD2_roi = imgcrop.readImageData()
#xdibias.imwrite(LOD2_roi, path_to_LOD + '/MUC_LoD2gt_Patch2', imgcrop)


'''
###############################################################################
#                                                                             #
#                         LOD2 Berlin                                         #
#                                                                             #
###############################################################################
'''

#path_to_LOD = '/home/davy_ks/la/Berlin'
#DSM = xdibias.Image('/sstore/data/davy_ks/cGAN/PlaneGAN/MultiTask_Paper/Dfilled_geoid_roi_xs_3489_ys_37512_width_500_height_350')
##DSM = xdibias.Image(path_to_LOD + '/DGM0.5m_smroi')
#
### Define bounding box
#bbox = DSM.boundingBox()
#print bbox.left,bbox.right, bbox.bottom, bbox.top
#
##Read image from which we will cut 
##LOD2 = xdibias.Image(path_to_LOD + '/LOD2_allBerlin/filledData/LOD2filled')
#LOD2 = xdibias.Image(path_to_LOD + '/LOD2filled_wth_DGM_roi_intrp')
##DGM = xdibias.Image(path_to_LOD + '/Dfilled_Berlin_roi')
#
#
#coord = np.array([[bbox.left,bbox.top]])
#pix = LOD2.map2pix(coord)
##pix = DGM.map2pix(coord)
#
##imgcrop = xdibias.Image(path_to_LOD + '/LOD2_allBerlin/filledData/LOD2filled_wth_DGM_roi_intrp', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#imgcrop = xdibias.Image(path_to_LOD + '/LOD2filled_wth_DGM_roi_intrp', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#LOD2_roi = imgcrop.readImageData()
#xdibias.imwrite(LOD2_roi, path_to_LOD + '/LOD2DSM_wth_DGM_roi_intrp_noRef_xs_3489_ys_37512_width_500_height_350', imgcrop)


'''
###############################################################################
#                                                                             #
#                         LOD2 Hamburg                                         #
#                                                                             #
###############################################################################
'''


#DSM = xdibias.Image("/home/davy_ks/project/pytorch/Coupled-cGAN/datasets/data_Washington/Dfilled/")
##DSM = xdibias.Image(path_to_LOD + '/DGM0.5m_smroi')

### Define bounding box
#bbox = DSM.boundingBox()
#print bbox.left,bbox.right, bbox.bottom, bbox.top

##Read image from which we will cut 
#DGM = xdibias.Image("/home/davy_ks/project/pytorch/Coupled-cGAN/datasets/data_Washington/Dstereo_num_mic/")
#coord = np.array([[bbox.left,bbox.top]])
#pix = DGM.map2pix(coord)

##imgcrop = xdibias.Image(path_to_LOD + '/LOD2_allBerlin/filledData/LOD2filled_wth_DGM_roi_intrp', xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#imgcrop = xdibias.Image("/home/davy_ks/project/pytorch/Coupled-cGAN/datasets/data_Washington/Dstereo_num_mic/", xs=int(pix[0][0]),ys=int(pix[0][1]),width=DSM.Columns,height=DSM.Rows)
#DGM_roi = imgcrop.readImageData()
##plt.show(plt.imshow(LOD2_roi[:1000,:1000]))
#mask = np.zeros((DGM_roi.shape), dtype=np.uint8)
#mask[DGM_roi!=0] = 1

#xdibias.imwrite(mask, "/home/davy_ks/project/pytorch/Coupled-cGAN/datasets/data_Washington/Dstereo_mask", imgcrop)