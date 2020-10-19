import os.path
import random
import yaml
import numpy as np
#import pylab as pl
import cv2 # new!
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data
#from data.base_dataset import BaseDataset
import xdibias
import pdb
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import torch.utils.data as data
from PIL import Image
import outlier_simulation as outsim
import logging
logger = logging.getLogger(__name__)


class XdibiasDSMLoader(data.Dataset):
    """Dataset for stereo DSM refinement to LiDAR/LoD2 DSM quality"""

    def __init__(self, opt, cfg, roi):
        """Initialize the training dataset"""

        self.opt = opt
        self.config = cfg
             
        if hasattr(opt,"DSM") and not opt.DSM is None: # skip
            self.config["data"]["DSM"] = opt.DSM[0]
            del self.config["data"]["Out"]

        if hasattr(opt,"Ortho") and  not opt.Ortho is None: # skip
            self.config["data"]["Ortho"] = opt.Ortho[0]
                
        """  load stereo DSM  """
        if len(roi) == 0:
            self.DSM = xdibias.Image(self.config["data"]["DSM"])
        else:
            self.DSM = xdibias.Image(self.config["data"]["DSM"],xs=int(roi[0]),ys=int(roi[1]),width=int(roi[2]),height=int(roi[3]))

        bbox = self.DSM.boundingBox() # set roi bounding box: xdibias.Image.boundingBox()        
        
        
        """  load ground truth DSM  """
        if "Out" in self.config["data"]:
            self.Out = xdibias.Image(self.config["data"]["Out"])
            assert(self.DSM.XCellRes == self.Out.XCellRes)
            assert(self.DSM.XCellRes == self.Out.XCellRes) ### Q7: YCellRes?
            xdibias.geo.intersectRect(bbox, self.Out.boundingBox())
        else:
            self.Out = None
            
        """  load orthophoto  """    
        if "Ortho" in self.config["data"]:
            # open ortho image and use that as second channel
            self.Ortho = xdibias.Image(self.config["data"]["Ortho"])
            assert(self.DSM.XCellRes == self.Ortho.XCellRes)
            assert(self.DSM.XCellRes == self.Ortho.XCellRes)
            xdibias.geo.intersectRect(bbox, self.Ortho.boundingBox())
        else:
            self.Ortho = None
        
        """  load building mask  """
        if "MASK" in self.config["data"]:
            # open mask image and use that as output for surface normals
            self.MASK = xdibias.Image(self.config["data"]["MASK"])
            assert(self.DSM.XCellRes == self.MASK.XCellRes)
            assert(self.DSM.XCellRes == self.MASK.XCellRes)
            xdibias.geo.intersectRect(bbox, self.MASK.boundingBox())
        else:
            self.MASK = None
        
        """  load core points (new!)  """
        if "Edges" in self.config["data"]:
            # open edges image and use that as ground truth for core_points prediction
            self.Edges = xdibias.Image(self.config["data"]["Edges"])
            assert(self.DSM.XCellRes == self.Edges.XCellRes)
            assert(self.DSM.XCellRes == self.Edges.XCellRes)
            xdibias.geo.intersectRect(bbox, self.Edges.boundingBox())
        
        
        
        """  cut roi  """
        # crop to common intersection and check Resolution
        #print "Union box", (bbox.left,bbox.right, bbox.bottom, bbox.top)
        self.DSM = self.DSM.getROIImage(bbox)
        if not self.Out is None:
            self.Out = self.Out.getROIImage(bbox, gridtol=0.5)
        if not self.Ortho is None:
            self.Ortho = self.Ortho.getROIImage(bbox, gridtol=0.5)
        if not self.MASK is None:
            self.MASK = self.MASK.getROIImage(bbox, gridtol=0.5)        
        if not self.Edges is None: 
            self.Edges = self.Edges.getROIImage(bbox, gridtol=0.5) # new!
                        
        # calculate number of training patches
        self.tilesPerRow = int(self.DSM.Columns / (self.opt.fineSize-self.opt.overlap)) # default: 30733/(256-0)=120
        self.ntiles = int(self.DSM.Rows / (self.opt.fineSize-self.opt.overlap)) * self.tilesPerRow # default: 40000/(256-0)*120=18720

        """  info to remove DSM outliers  """
        # compute normalization using DSM
        # normalize with 5% quantiles.
        # Extract minQuantile from config file
        # Compute minQuant for all training area
        # It removes outliers 
        if self.opt.isTrain == True: 
            IN = self.DSM.readImageData()
            self.minQuant = np.percentile(IN,self.config["minQuantile"]) # 1% of DSM height values, default: 26
            
            ## compute outliers for modeling
            self.data_min = IN.min() # default: -41
            self.data_max = IN.max() # default: 308
            
            # save to the disk train_options
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            d = {'value_minQuant': str(self.minQuant), 
                 'minQuantile': str(self.config["minQuantile"])}
                 
            if self.Ortho:
                self.p2, self.p98 = np.percentile(self.Ortho.readImageData(), (1, 99)) # 2% and 98% of orthophoto values, default: 102,475
                d = {'p2' : str(self.p2), 
                     'p98': str(self.p98)}
                 
            with open((os.path.join(expr_dir, "train_config.yaml")), 'w') as cf:
                yaml.dump(d, cf, default_flow_style=False) ### Q3: save ortho only?
                
        # Test phase    
        else:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
#            with open(os.path.join(expr_dir, "train_config.yaml")) as cf:
#                self.test_config = yaml.load(cf)

#            # Berlin city             
#            self.minQuant = float(self.test_config["value_minQuant"])                           
#            self.p2 = float(self.test_config["p2"])
#            self.p98 = float(self.test_config["p98"])

            
            # NEW CITY            
            IN = self.DSM.readImageData()
            #self.minQuant = np.nanpercentile(IN,self.config["minQuantile"]) # DSM percentile for new city (if data contains nan values)
            self.minQuant = np.percentile(IN,self.config["minQuantile"]) 
            
            if self.Ortho:
                self.p2, self.p98 = np.percentile(self.Ortho.readImageData(), (1, 99))
                print "self.p2, self.p98:", self.p2, self.p98
        
        """  transform to tensor  """    
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
           

    def __getitem__(self, index):
    
        """  position of this patch  """
        # compute tile position
        j = (index % self.tilesPerRow) * (self.opt.fineSize-self.opt.overlap)
        i = (index / self.tilesPerRow) * (self.opt.fineSize-self.opt.overlap)

        if self.opt.isTrain:
            # slight random shifting of the tile, up to one tilesize
            j = random.randint(j, max(j, min(j+self.opt.fineSize, self.DSM.Columns - self.opt.fineSize - 1)))
            i = random.randint(i, max(i, min(i+self.opt.fineSize, self.DSM.Rows - self.opt.fineSize - 1)))
        else:
            #special case for last tile      
            i = min(i, self.DSM.Rows - self.opt.fineSize)
            j = min(j, self.DSM.Columns - self.opt.fineSize)
            
            print i, j
        
        #pdb.set_trace()
        """  extract patch img from dataset  """        
        ## stereo dsm ##
        input_dsm, norm_params = self.getPatch(self.DSM, i, j, norm=True, outliers = False) ### outliers = self.opt.isTrain Q4: not using outliers?
        #print "input_dsm", input_dsm.min(), input_dsm.max()
        input_dsm = Image.fromarray(input_dsm) # stereo dsm
        
        ## ground truth dsm ##
        if self.Out is not None:       
            gt_model = self.getPatch(self.Out, i, j, norm=norm_params) # share stereo dsm's norm_param
            #print "gt_model", gt_model.min(), gt_model.max()
            gt_model = Image.fromarray(gt_model)

        ## ortho photo ##
        if self.Ortho is not None:
            input_ortho = self.getPatch(self.Ortho, i, j, norm="intensity") # norm p2,p98 & [0,255]
            #print "input_ortho", input_ortho.min(), input_ortho.max()
            input_ortho = Image.fromarray(input_ortho)

        ## building mask ##    
        if self.MASK:
            gt_mask = self.getPatch(self.MASK, i, j, norm=False)
            gt_mask = Image.fromarray(gt_mask)
        
        ## Edges (new!) ##
        if self.Edges:
            pdb.set_trace()
            gt_edges = self.getPatch(self.Edges, i, j, norm=False)
            #gt_edges = Image.fromarray(gt_edges)
            
                
        """  random flip  """    
        if self.opt.isTrain:    
            # Random horizontal flipping
            if random.random() > 0.5:            
                input_dsm = TF.hflip(input_dsm)
                gt_model = TF.hflip(gt_model)
                if self.Ortho is not None:
                    input_ortho = TF.hflip(input_ortho)
                if self.MASK is not None:
                    gt_mask = TF.hflip(gt_mask)
                if self.Edges is not None:
                    gt_edges = cv2.flip(gt_edges,1) # new!
                           
            if random.random() < 0.5:            
                input_dsm = TF.vflip(input_dsm)
                gt_model = TF.vflip(gt_model)
                if self.Ortho is not None:
                    input_ortho = TF.vflip(input_ortho)
                if self.MASK is not None:
                    gt_mask = TF.vflip(gt_mask)
                if self.Edges is not None:
                    gt_edges = cv2.flip(gt_edges,0) # new!    
                        
        pdb.set_trace()
        """  create return samples  """
        sample = {} # create return sample
           
        sample["A"] = self.transform(input_dsm) # A: stereo dsm
        sample["B"] = self.transform(gt_model) # B: ground truth dsm
        if self.Ortho is not None:
            sample["Ortho"] = self.transform(input_ortho) # Ortho: orthophoto
        if self.MASK is not None:
            sample["M"] = self.transform(gt_mask) # M: building mask
        if self.Edges is not None:
            sample["Edges"] = self.transform(gt_edges) # Edges: building edges (new!)
                
        sample["factor"] = norm_params # factor: norm params of stero dsm
       
        return sample
        
    def getPatch(self, img, i, j, norm=False, outliers = False):
        """Get a patch from a raster image and optionally normalize it"""

        patch = img.readImageData(roi=(slice(i,i+self.opt.fineSize),
                                       slice(j,j+self.opt.fineSize)))
        #pdb.set_trace()                               
        if outliers and self.opt.isTrain: ### Q5: outliers=False? 
            if random.random() > 0.5: ### Q6: random?
                patch = outsim.outlier_simulation(patch, self.data_min,self.minQuant) 
                #pdb.set_trace()

        limits = None
        if norm:
            if type(norm) == bool:
                patch, limits = self.normalize(patch, zero_center=True,
                                                      return_limits=True)
            elif type(norm) == str:
                patch = self.intensity_rescaling(patch)
                patch = self.normalize(patch.astype(np.float32), limits = (0.0, 255.0), zero_center=True,
                                                      return_limits=False)
                
            else:
                patch = self.normalize(patch, limits=norm,
                                              zero_center=True)

        if limits is not None:
            return patch, limits
        else:
            return patch


    def normalize(self, img, limits=None, zero_center=False,
                         return_limits=False, restore=False):

        # run in reverse (undo scaling)
        if restore:
            if zero_center:
                img /= 2
                img += 0.5
            img *= (limits[1] - limits[0])
            img += limits[0]
            return img

        # run forward (scale intensities)
        else:
            if limits is None:
                limits = (img.min(), img.max())
            img -= limits[0]
            img /= (limits[1] - limits[0])
            if zero_center:
                img -= 0.5
                img *= 2
            if return_limits:
                return img, limits
            else:
                return img

    def intensity_rescaling(self, image):
        out = np.zeros_like(image)
        if len(image.shape) > 2:
            for i in range (0,image.shape[-1]):
                out[:,:,i] = (image[:,:,i]-self.p2)*255/(self.p98-self.p2)
            out[out<0] = 0
            out[out>255] = 255
        else:
            out = (image-self.p2)*255/(self.p98-self.p2)
            out[out<0] = 0
            out[out>255] = 255           
        return out.astype(np.uint8)  


            
    def __len__(self):
        return self.ntiles

    def name(self):
        return 'XdibiasDSMLoader'
