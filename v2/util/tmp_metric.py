import xdibias
import numpy as np
import pdb
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import shapefile
import skimage.draw
import cv2
import math
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score,recall_score

def RMSE(prediction, label):
    #prediction[np.abs(prediction-label)>25] = label[np.abs(prediction-label)>25]
    return np.sqrt(np.mean((prediction - label)**2))

def MAE(prediction, label):
    #prediction[np.abs(prediction-label)>25] = label[np.abs(prediction-label)>25]
    return np.mean(np.abs(prediction-label))

def NMAD(prediction, label):
    #pdb.set_trace()
    diff = prediction-label
    median_d = np.median(diff)
    median_m = np.median(np.abs(diff-median_d))
    return 1.4826 * median_m


SMOOTH = 1e-6
def IoU(prediction, label):
    intersection = (prediction & label).sum()
    union = (prediction | label).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou



#####  DSM: RMSE, MAE, NMAD  #####

dsm_stereo = xdibias.Image('/home/wang_yi/la/dataset/WV-1/Dfilled_geoid_roi')

#dsm_pred = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/uresnet_dsm')          # urn_dsm
dsm_pred = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/cgan_dsm_sl1')         # urn_dsm_gan
#dsm_pred = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/interp_dsm')           # cgan_dsm_interp
#dsm_pred = xdibias.Image('/home/wang_yi/la/UResNet-only/test_output/unet_only/result_dsm') # urn_dsm_edge

#dsm_pred = xdibias.Image('/home/wang_yi/la/cGAN-bm/test_output/urn_mt_1/result_dsm')       # urn_mask_nDSM

#dsm_pred = xdibias.Image('/home/wang_yi/la/cGAN-hed/test_output/urn_mt_2/result_dsm')      # urn_mt_hed

#dsm_pred = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/cgan_dsm_w0')             # cgan_dsm_w0
#dsm_pred = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/result_dsm')             # cgan_dsm_w0_fine
#dsm_pred = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/result_dsm_all')             
#dsm_pred = xdibias.Image('/home/wang_yi/la/cGAN-light/test_output/cgan_dsm_ortho')             # cgan_ndsm_ortho

#dsm_pred = xdibias.Image('/home/wang_yi/la/cGAN-light/result_dsm')  # w_ganloss = -2.7
#dsm_pred = xdibias.Image('/home/wang_yi/la/dsm_refinement/result_dsm') # w_ganloss=1,w_l1=20,w_sn=4
#dsm_pred = xdibias.Image('/home/wang_yi/la/Coupled-cGAN/test_output/result_dsm') 


#dsm_pred = xdibias.Image('/home/wang_yi/la/test_cgan_multi_loss/test_output/result_dsm')

#dsm_pred = xdibias.Image('/home/wang_yi/la/attn/test_output/attn_1/result_dsm')
#dsm_pred = xdibias.Image('/home/wang_yi/la/attn/test_output/attn_4/result_dsm')
#dsm_pred = xdibias.Image('/home/wang_yi/la/attn_dsm/test_output/attn_dsm/result_dsm')

dsm_ks = xdibias.Image('/home/wang_yi/la/temp/version0/result_Coupled_UResNet50_ksenia')
dsm_gt = xdibias.Image('/home/wang_yi/la/dataset/LOD2filled_wth_DGM_roi_intrp_GEOKOLref')
#dsm_gt = xdibias.Image('/home/wang_yi/la/dataset/nLOD2filled_wth_DGM_roi_intrp_GEOKOLref')

bbox = dsm_ks.boundingBox()

stereo_roi = dsm_stereo.getROIImage(bbox,gridtol=0.5)
data_stereo = stereo_roi.readImageData()[20500:21000,1500:2000]

pred_roi = dsm_pred.getROIImage(bbox,gridtol=0.5)
data_pred = pred_roi.readImageData()[20500:21000,1500:2000]

data_ks = dsm_ks.readImageData()[20500:21000,1500:2000]

gt_roi = dsm_gt.getROIImage(bbox,gridtol=0.5)
data_gt = gt_roi.readImageData()[20500:21000,1500:2000]


rmse_stereo = RMSE(data_stereo, data_gt)
mae_stereo = MAE(data_stereo, data_gt)
nmad_stereo = NMAD(data_stereo, data_gt)

rmse_pred = RMSE(data_pred, data_gt)
mae_pred = MAE(data_pred, data_gt)
nmad_pred = NMAD(data_pred, data_gt)

rmse_ks = RMSE(data_ks,data_gt)
mae_ks = MAE(data_ks, data_gt)
nmad_ks = NMAD(data_ks, data_gt)

print '## refined DSM ##'
print 'RMSE, MAE, NMAD'
print 'metric_stereo: ', rmse_stereo, mae_stereo, nmad_stereo
print 'metric_pred: ', rmse_pred, mae_pred, nmad_pred
#print 'metric_ks: ', rmse_ks, mae_ks, nmad_ks
print ' '

#pdb.set_trace()

#####  3D model: footprint  #####
mask_pred = xdibias.Image('/home/wang_yi/la/temp/version0/results_roi_xdb/mask_out_roi')
mask_gt = xdibias.Image('/home/wang_yi/la/temp/version0/results_roi_xdb/mask_gt_roi')

data_mask_pred = mask_pred.readImageData()[500:1000,500:1000]
data_mask_gt = mask_gt.readImageData()[500:1000,500:1000]

#pdb.set_trace()

#precision_mask = precision_score(data_mask_gt.flatten(),data_mask_pred.flatten())
#recall_mask = recall_score(data_mask_gt.flatten(),data_mask_pred.flatten())
#f1_score = 2*precision_mask*recall_mask/(precision_mask+recall_mask)

accuracy_mask = accuracy_score(data_mask_gt.flatten(),data_mask_pred.flatten())
#IoU_mask = jaccard_score(data_mask_gt.flatten(),data_mask_pred.flatten())
IoU_mask = IoU(data_mask_pred.flatten().astype('uint8'),data_mask_gt.flatten().astype('uint8'))

print '## footprint ##'
print 'accuracy: ', accuracy_mask
print 'IoU: ', IoU_mask
print ' '


#####  3D model: ridge&eave height difference  #####
ndsm_out = xdibias.Image('/home/wang_yi/la/temp/version0/results_roi_xdb/ndsm_out_roi') 
ndsm_gt = xdibias.Image('/home/wang_yi/la/temp/version0/results_roi_xdb/ndsm_gt_roi')

data_ndsm_out = ndsm_out.readImageData()[500:1000,500:1000]
data_ndsm_gt = ndsm_gt.readImageData()[500:1000,500:1000]
2.1244144
mask = data_mask_pred.astype('uint8')
kernel = np.ones((7,7),np.uint8)
mask_eroded = cv2.erode(mask,kernel,iterations = 1)
#pdb.set_trace()
height_out = data_ndsm_out[mask_eroded==1]
height_gt = data_ndsm_gt[mask_eroded==1]

rmse_h = RMSE(height_out, height_gt)
mae_h = MAE(height_out, height_gt)
nmad_h = NMAD(height_out, height_gt)

print '## 3D model: height ##'
print 'RMSE, MAE, NMAD'
print 'filtered ndsm: ', rmse_h, mae_h, nmad_h
print ' '

#pdb.set_trace()

#####  3D model: orientation  #####
# planarity_evaluation.py

GT_depth = xdibias.Image('/home/wang_yi/la/temp/version0/results_roi_xdb/dsm_gt_roi')
GT_DEPTH = GT_depth.readImageData()

sf = shapefile.Reader("/home/wang_yi/la/temp/version0/results_roi/sample_polys_in_geo.shp")
depth = xdibias.Image('/home/wang_yi/la/temp/version0/results_roi_xdb/dsm_roi') 
DEPTH = depth.readImageData()

fields = sf.fields[1:]
field_names = [field[0] for field in fields]
buffer = []
   
for sr in sf.shapeRecords():
   atr = dict(zip(field_names, sr.record))
   geom = sr.shape.__geo_interface__
   buffer.append(dict(geometry=geom))

x = []
y = []
z = []

flat = 0
orie = 0
orie_arr = []

skipped = 0

for num_geom in range(len(buffer)):

    pix = 0
    x = []
    y = []
    z = []
    
    for point in buffer[num_geom]['geometry']['coordinates'][0]:
        coord = np.array([point])
        pix = depth.map2pix(coord)
        
        x.append(pix[0][0])
        y.append(pix[0][1])
        z.append(DEPTH[int(pix[0][1]),int(pix[0][0])])
    
    
    mask = np.zeros([depth.Rows, depth.Columns], dtype=np.uint8)
    #pdb.set_trace()
    rr, cc = skimage.draw.polygon(y, x)
    mask[rr, cc] = 1
    
    kernel = np.ones((7,7),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    
    #plt.show(plt.imshow(mask))
    #plt.show(plt.imshow(erosion))
    
    indices = np.where(erosion == [1])
    
    # only consider plane masks which are bigger than 5% of the image dimension
    if len(indices[0]) <= 2:
        skipped += 1

    else:
      
        zz = DEPTH[indices]
                
        pointCloud = np.stack((indices[1], indices[0], zz))

        # fit 3D plane to 3D points (normal, d)
        pca = PCA(n_components=3)
        pca.fit(pointCloud.T)
        normal = -pca.components_[2,:] 
        point = np.mean(pointCloud,axis=1)
        d = -np.dot(normal,point);
        
        #print "normal", normal
        # PE_flat: deviation of fitted 3D plane
        flat += np.std(np.dot(pointCloud.T,normal.T)+d)#*100.
        
        #print "flat: ", np.std(np.dot(pointCloud.T,normal.T)+d) #*100.
        
        
        GT_zz = GT_DEPTH[indices]
        GT_pointCloud = np.stack((indices[1], indices[0], GT_zz))
        # fit 3D plane to 3D points (normal, d)
        gt_pca = PCA(n_components=3)
        gt_pca.fit(GT_pointCloud.T)
        gt_normal = -gt_pca.components_[2,:] 
        gt_point = np.mean(GT_pointCloud,axis=1)
        gt_d = -np.dot(gt_normal,gt_point);
        
        if np.dot(normal,gt_normal)<0:
           normal = -normal
        
        # PE_ori: 3D angle error between ground truth plane and normal vector of fitted plane
        tmp_orie = math.atan2(np.linalg.norm(np.cross(gt_normal,normal)),np.dot(gt_normal,normal))*180./np.pi 
        orie += tmp_orie
        orie_arr.append(tmp_orie)
                   
        #print "orie: ", math.atan2(np.linalg.norm(np.cross(gt_normal,normal)),np.dot(gt_normal,normal))*180./np.pi

print '## flatness & orientation ##'
print "avg_flat", flat/(len(buffer)-skipped)        
print "avg_orie", orie/(len(buffer)-skipped)


np_orie = np.asarray(orie_arr)
avg_orie = np_orie.mean()
min_orie = np_orie.min()
max_orie = np_orie.max()
sigma_orie = np_orie.std()
print "avg_orie", avg_orie, "min_orie", min_orie, "max_orie", max_orie, "sigma_orie", sigma_orie
     
print ' '








