import xdibias
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score,recall_score,precision_recall_curve
import torch
import pdb
import matplotlib.pyplot as plt
import cv2


def precision_recall_curve2(y_true,y_pred,thresholds):
    precision = []
    recall = []
    #pdb.set_trace()
    for i,threshold in enumerate(thresholds):
        y_pred_i = np.zeros(y_pred.shape)
        y_pred_i[y_pred>threshold] = 1
        precision_i = precision_score(y_true,y_pred_i)
        recall_i = recall_score(y_true,y_pred_i)
        
        precision.append(precision_i)
        recall.append(recall_i)
    return precision,recall
    

roi_row1 = 20400
roi_row2 = 21400
roi_col1 = 1800
roi_col2 = 2800

##########  pred  ############
#test_img = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/unet_only/result_edges')
#test_img = xdibias.Image('/home/wang_yi/la/Coupled-cGAN/test_output/result_edges')

#test_img = xdibias.Image('/home/wang_yi/la/UNet_edge2/test_output/dsm_mt/result_edges')
#test_img = xdibias.Image('/home/wang_yi/la/UNet_edge2/test_output/dsm_ks/result_edges')
#test_img = xdibias.Image('/home/wang_yi/la/unet_edge/test_output/urn_edge_hed/result_edges')

#test_img = xdibias.Image('/home/wang_yi/la/cGAN-bm/test_output/urn_mt_1/result_edges')  # fail

#test_img = xdibias.Image('/home/wang_yi/la/cGAN-hed/test_output/urn_mt_2/result_edges') # bad

#test_img = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/cgan_mt_ortho/result_edges')

#test_img = xdibias.Image('/home/wang_yi/la/attn/test_output/attn_4/result_edges')
test_img = xdibias.Image('/home/wang_yi/la/attn_edge/test_output/attn_edge/result_edges')

data = test_img.readImageData()

data0 = data[roi_row1:roi_row2,roi_col1:roi_col2,:]
data1 = data0
#pdb.set_trace()
#data1 = torch.nn.Softmax(dim=-1)(torch.tensor(data0)).data.numpy()
#data2 = data1.copy()
data2 = np.zeros_like(data1)

data2[:,:,2] = 0.0
data2[:,:,1] = 0.0
data2[data1[:,:,1]>=0.15,1]=1.0
data2[data1[:,:,1]<0.15,1]=0.0
data2[data1[:,:,0]>=0.05]=(1.0,0.0,0.0)
#data2[data1[:,:,0]<0.7]=(1.0,0.0,0.0)
#data2[data1[:,:,0]<0.1,0]=0.0





#pdb.set_trace()
img = Image.fromarray((data2*255.0).astype('uint8'))
img.save('sample_edge.png')
#img.save('sample_cGAN_0.png')
print 'pred result saved.'

#pdb.set_trace()
#########  gt  ###########
gt_img = xdibias.Image('/home/wang_yi/la/dataset/edge_gt')
bbox = test_img.boundingBox()
xdibias.geo.intersectRect(bbox,gt_img.boundingBox())
gt_roi = gt_img.getROIImage(bbox,gridtol=0.5)

data_gt = gt_roi.readImageData()
data3 = data_gt[roi_row1:roi_row2,roi_col1:roi_col2]

data4 = np.zeros(data2.shape)
data4[data3==1]=(0.0,1.0,0.0)
data4[data3==2]=(1.0,0.0,0.0)

img2 = Image.fromarray((data4*255.0).astype('uint8'))
img2.save('sample_gt.png')
print 'gt result saved.'

##########  ortho  ###########
ortho_img = xdibias.Image('/home/wang_yi/la/dataset/WV-1/O.3')
xdibias.geo.intersectRect(bbox,ortho_img.boundingBox())
ortho_roi = ortho_img.getROIImage(bbox,gridtol=0.5)
data_ortho = ortho_roi.readImageData()
data5 = data_ortho[roi_row1:roi_row2,roi_col1:roi_col2]
p2,p98 = np.percentile(data5, (1, 99))
data6 = (data5-p2)/(p98-p2)
#data6 = data5.copy()
data6[data6<0] = 0
data6[data6>1] = 1
#print p2,p98
#pdb.set_trace()
img3 = Image.fromarray((data6*255).astype('uint8'))
#img3 = Image.fromarray(data6)
img3.save('sample_ortho.png')
print 'ortho result saved.'


#########  dsm  ###########
#dsm_img = xdibias.Image('/home/wang_yi/la/dataset/WV-1/Dfilled_geoid_roi')
#dsm_img = xdibias.Image('/home/wang_yi/la/attn/test_output/attn_4/result_dsm')
dsm_img = xdibias.Image('/home/wang_yi/la/attn_dsm/test_output/attn_dsm/result_dsm')
#dsm_img = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/cgan_mt_ortho/result_dsm')
xdibias.geo.intersectRect(bbox,dsm_img.boundingBox())
dsm_roi = dsm_img.getROIImage(bbox,gridtol=0.5)

data_dsm = dsm_roi.readImageData()
data7 = data_dsm[roi_row1:roi_row2,roi_col1:roi_col2]

p2_d,p98_d = np.percentile(data7, (1, 99))
data8 = (data7-p2_d)/(p98_d-p2_d)
data8[data8<0] = 0
data8[data8>1] = 1

img3 = Image.fromarray((data8*255).astype('uint8'))
img3.save('sample_dsm.png')

#img4 = Image.fromarray(data7)
#img4.save('sample_dsm_raw.tif')

print 'dsm result saved.'

'''
#######  dsm_kse  ##########
kdsm_img = xdibias.Image('/home/wang_yi/la/temp/result_Coupled_UResNet50_ksenia')
xdibias.geo.intersectRect(bbox,kdsm_img.boundingBox())
kdsm_roi = kdsm_img.getROIImage(bbox,gridtol=0.5)

data_kdsm = kdsm_roi.readImageData()
data9 = data_kdsm[roi_row1:roi_row2,roi_col1:roi_col2]

p2_d,p98_d = np.percentile(data7, (1, 99))
data10 = (data9-p2_d)/(p98_d-p2_d)
data10[data10<0] = 0
data10[data10>1] = 1

img3 = Image.fromarray((data10*255).astype('uint8'))

img3.save('sample_kdsm.png')
print 'kdsm result saved.'
'''

#######  dsm_gt  ##########
gtdsm_img = xdibias.Image('/home/wang_yi/la/dataset/LOD2filled_wth_DGM_roi_intrp_GEOKOLref')
xdibias.geo.intersectRect(bbox,gtdsm_img.boundingBox())
gtdsm_roi = gtdsm_img.getROIImage(bbox,gridtol=0.5)

data_gtdsm = gtdsm_roi.readImageData()
data9 = data_gtdsm[roi_row1:roi_row2,roi_col1:roi_col2]

p2_d,p98_d = np.percentile(data7, (1, 99))
data10 = (data9-p2_d)/(p98_d-p2_d)
data10[data10<0] = 0
data10[data10>1] = 1

img3 = Image.fromarray((data10*255).astype('uint8'))

img3.save('sample_gtdsm.png')
print 'gtdsm result saved.'


#######  dsm_stereo  ##########
stdsm_img = xdibias.Image('/home/wang_yi/la/dataset/WV-1/Dfilled_geoid_roi')
xdibias.geo.intersectRect(bbox,stdsm_img.boundingBox())
stdsm_roi = stdsm_img.getROIImage(bbox,gridtol=0.5)

data_stdsm = stdsm_roi.readImageData()
data9 = data_stdsm[roi_row1:roi_row2,roi_col1:roi_col2]

p2_d,p98_d = np.percentile(data7, (1, 99))
data10 = (data9-p2_d)/(p98_d-p2_d)
data10[data10<0] = 0
data10[data10>1] = 1

img3 = Image.fromarray((data10*255).astype('uint8'))

img3.save('sample_stdsm.png')
print 'stdsm result saved.'







#####  recall  ######
data2[:,:,2] = 0.6
data4[:,:,2] = 0.6
y_true = np.argmax(data4,axis=-1)
y_pred = np.argmax(data2,axis=-1)

#pdb.set_trace()
#recall = recall_score(y_true.flatten(),y_pred.flatten(),average=None)
recall = recall_score(y_true.flatten(),y_pred.flatten(),average=None)
print 'recall score: ', recall


'''
###  precision_recall_curve  ###
y_true_c = data4[:,:,0].flatten()
y_pred_c = data1[:,:,0].flatten()
#precision_c,recall_c,thresholds_c = precision_recall_curve(y_true_c,y_pred_c)
#pdb.set_trace()
thresholds_c = np.linspace(0,1,21)
precision_c,recall_c = precision_recall_curve2(y_true_c,y_pred_c,thresholds_c)

print '#######################################'
print '\nprecision_c: ',precision_c
print '\nrecall_c: ',recall_c
print '\nthresholds_c: ',thresholds_c

y_true_e = data4[:,:,1].flatten()
y_pred_e = data1[:,:,1].flatten()
#precision_e,recall_e,thresholds_e = precision_recall_curve(y_true_e,y_pred_e)
thresholds_e = np.linspace(0,1,21)
precision_e,recall_e = precision_recall_curve2(y_true_e,y_pred_e,thresholds_e)

print '######################################'
print '\nprecision_e: ',precision_e
print '\nrecall_e: ',recall_e
print '\nthresholds_e: ',thresholds_e
'''

'''
#fig1 = plt.figure()
plt.plot(recall_c,precision_c,recall_e,precision_e)
plt.legend(['corner','edge'])
plt.xlabel('recall')
plt.ylabel('precision')
#plt.savefig('prec_rec_cGAN_0.png')
plt.savefig('prec_rec_unet_edge.png')
#plt.close(fig1)
'''
