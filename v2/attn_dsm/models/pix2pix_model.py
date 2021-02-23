import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
#from . import networks as networks
from . import networks as networks
import pdb
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import itertools
import sys
#sys.path.append('/home/davy_ks/project/pytorch/GDLossGAN/')
#from torchviz import make_dot, make_dot_from_trace
from torch.nn import init
from loss_functions import FullNormalLoss

#from robust_loss_pytorch import AdaptiveLossFunction

import logging
logger = logging.getLogger(__name__)

"""  function: display image  """
def show_grayscale_image(tensor):
    # IPython.display can only show images from a file.
    # So we mock up an in-memory file to show it.
    # IPython.display needs a numpy array with channels first.
    # and it also has to be uint8 with values between 0 and 255.

    # Make sure the images are still between 0 and 1.
    tensor = (tensor - tensor.min()).div(tensor.max() - tensor.min())
        
    f = BytesIO()
    a = np.uint8(tensor.mul(255).numpy()) 
    img = Image.fromarray(a)
    plt.show(plt.imshow(img))

"""  function: initialize weights  """    ### Q8: usage??
def weights_init(m):
    classname = m.__class__.__name__
    #pdb.set_trace()
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

"""  function: initialize weights  """    ### Q8: usage???
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

"""  class: Pix2PixModel(BaseModel)  """        
class Pix2PixModel(BaseModel):

    """ model name  """
    def name(self):
        return 'Pix2PixModel'

    """ model initialization """
    def initialize(self, opt):
    
        BaseModel.initialize(self, opt) # opt, gpu_ids, isTrain, Tensor, save_dir
        #pdb.set_trace()
        #self.isTrain = opt.isTrain ### Q9: repeat??                           
        self.input = {}
                   
        if self.isTrain:
            pretrained = True
        else:
            pretrained = False
        
        """ load generator and discriminator """
        
        """ Generator """    
        list_of_networks = ['Coupled_unet256', "DeepLab", "Coupled_UResNet", 'DeepLabv3_plus', 
                            'Corentin_UNetResNet50', 'Single_UResNet', "Single_R2U_Net", 
                            "R2AttU_Net","Coupled_R2U_Net", 'Coupled_UResNet18','Coupled_UResNet34', 
                            'Coupled_UResNet50', 'Coupled_UResNet101','Coupled_UResNet152',"PConvUResNet50", "Single_PConvUNet"]
                            
        to_freeze = ["Coupled_UResNet", 'DeepLabv3_plus','Single_UResNet', "DeepLab"] ### Q10: usage??
        
        if opt.which_model_netG != "W_GAN": # default: Coupled_UResNet50
            #pdb.set_trace()
            self.netG = networks.define_G(opt.input_nc, 
                                          opt.output_nc, 
                                          opt.ngf,
                                          opt.which_model_netG, 
                                          opt.norm, # batch Q10: ever tried instance?
                                          not opt.no_dropout, 
                                          self.gpu_ids, 
                                          pretrained,  
                                          opt.output_func, 
                                          opt.fusion,
                                          opt.task)  ## new!
                                                                                  
        else:    
            #opt.output_nc = 64
            netG1 = networks.define_G(opt.input_nc, 
                                          opt.output_nc, 
                                          opt.ngf,
                                          opt.which_model_netG, 
                                          opt.norm, 
                                          not opt.no_dropout, 
                                          self.gpu_ids, 
                                          pretrained,  
                                          opt.output_func, 
                                          opt.fusion,
                                          opt.task)
            
            netG2 = networks.define_G(opt.input_nc, 
                                          opt.output_nc, 
                                          opt.ngf,
                                          opt.which_model_netG, 
                                          opt.norm, 
                                          not opt.no_dropout, 
                                          self.gpu_ids, 
                                          pretrained,  
                                          opt.output_func, 
                                          opt.fusion,
                                          opt.task)

                                          
            print "number of self.netG1 param: ", len(list(netG1.parameters()))       
            print "number of self.netG2 param: ", len(list(netG2.parameters())) 
               
            self.netG = networks.concatNet(netG1,netG2, opt.input_nc, opt.output_nc, opt.output_func)  ### Q11: W_GAN --- two generators?
                          
        print "number of self.netG param: ", len(list(self.netG.parameters()))
        
        if len(self.gpu_ids) > 0:
            self.netG.cuda(self.gpu_ids[0])
        
        """ initialize or load weights """
        # if "start training" 
        if self.isTrain:
            
            if opt.which_model_netG == "Single_R2U_Net": 
                init_weights(self.netG, init_type='kaiming')
                
            elif opt.which_model_netG not in list_of_networks:
                self.netG.apply(weights_init) ### ????
            
            """ Discriminator """    
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)            
                  
        
        # if "testing or continue training"                 
        if not self.isTrain or opt.continue_train: # default: continue_train=False
            # load generator
            self.load_network(self.netG, 'G', opt.which_epoch)
            
            if opt.which_model_netG in to_freeze: # Q: freeze BatchNorm2d???
                self.netG.freeze_bn() 
            
            if self.isTrain:
                # load discriminator
                self.load_network(self.netD, 'D', opt.which_epoch)

                # if learning with learned loss weights
                # load learning weights
                if self.opt.loss_weights:
                    save_filename = '%s_net_%s.pth' % (opt.which_epoch, "task_weights")
                    save_path = os.path.join(self.save_dir, save_filename)
                    self.task_weights = torch.load(save_path)                    
        
        """ define losses """
        if self.isTrain:
            
            self.fake_AB_pool = ImagePool(opt.pool_size) # default: pool_size=0   ???
            self.old_lr = opt.lr
            
            # define loss functions
            """ 01: GAN loss (generator loss?) """
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
             
            """ 02: L1 loss """
            self.criterionL1 = torch.nn.L1Loss()
                
            """ 03: normal loss """ ## to be learned
            if self.opt.lambda_SN !=0.0:                            
                #self.criterionSN = networks.SurfNormalLoss(tensor=self.Tensor)
                if self.opt.which_Reg == 'NormalLoss':
                    self.criterionSN = networks.NormalLoss(tensor=self.Tensor) 
                elif self.opt.which_Reg == "FullNormalLoss":
                    self.criterionSN = FullNormalLoss() 
                else:
                    self.criterionSN = networks.BuidingSurfaceNormalLoss(tensor=self.Tensor)
            
            # new!!
            if self.opt.task == 'dsm_edges':
                self.criterionEdges = torch.nn.CrossEntropyLoss()
                
                
            """ weights of the losses & optimizers """
            if self.opt.loss_weights:
                #pdb.set_trace()
                if not opt.continue_train:
                    # set up losses learning weights
                    self.task_weights = {}
                    # weighting parameter
                    # new!!
                    init_w = 0.5
                    if self.opt.task == 'dsm_edges':
                        init_w = 0.33
                        self.LE = torch.nn.Parameter(torch.as_tensor(np.log(init_w**2)).to("cuda"),requires_grad=True)
                    
                    self.L1 = torch.nn.Parameter(torch.as_tensor(np.log(init_w ** 2)).to("cuda"), requires_grad=True)
                    self.SN = torch.nn.Parameter(torch.as_tensor(np.log(init_w ** 2)).to("cuda"), requires_grad=True)
                    self.GAN = torch.nn.Parameter(torch.as_tensor(np.log(init_w ** 2)).to("cuda"), requires_grad=False)
                        
                    if self.opt.task == 'dsm':
                        self.task_weights = {"L1": self.L1, "SN": self.SN, 'GAN':self.GAN}
                    elif self.opt.task == 'dsm_edges':
                        #self.task_weights = {"L1": self.L1, "SN": self.SN, "LE": self.LE}
                        self.task_weights = {"L1": self.L1, "SN": self.SN, 'GAN':self.GAN, "LE": self.LE}
                    
                else:
                    self.task_weights["L1"] = torch.nn.Parameter(torch.as_tensor(self.task_weights["L1"]).to("cuda"), requires_grad=True)
                    self.task_weights["SN"] = torch.nn.Parameter(torch.as_tensor(self.task_weights["SN"]).to("cuda"), requires_grad=True)
                    self.task_weights["GAN"] = torch.nn.Parameter(torch.as_tensor(self.task_weights["GAN"]).to("cuda"), requires_grad=False)
                    if self.opt.task == 'dsm_edges':
                        self.task_weights["LE"] = torch.nn.Parameter(torch.as_tensor(self.task_weights["LE"]).to("cuda"), requires_grad=True)
                
                # initialize generator optimizers
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), list(self.task_weights.values())),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))                  
                                                     
            else:
                # initialize generator optimizers
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            # initialize discriminator optimizers
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        """ network initialization finished """
        #self.netG.eval()
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')
        
        """ store network sturcture """
        if self.isTrain:
          if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
              os.makedirs(os.path.join(opt.checkpoints_dir, opt.name))
                       
          f = open( os.path.join(opt.checkpoints_dir, opt.name) + '/netG.txt', 'w' )
          f.write( str(self.netG) )
          f.write( str(self.netD) )
          f.close()
          #quit()
        else:
          test_outpath = os.path.join(opt.test_outdir,opt.Out[0]) # new!
          if not os.path.exists(test_outpath):
              os.makedirs(test_outpath)                     
          f = open( test_outpath + '/net.txt', 'w' )
          f.write( str(self.netG) )
          f.close()
          
    """  set inputs  """
    def set_input(self, input):
        self.input = []
        self.idict = input
        
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']        
        self.input_A = self.Tensor(input_A.size()).copy_(input_A)
        self.input_B = self.Tensor(input_B.size()).copy_(input_B)

        self.input.append(input_A)
        
        if "Ortho" in input:
            input_O = input['Ortho']
            self.input_O = self.Tensor(input_O.size()).copy_(input_O)
            self.input.append(input_O)
            
        if 'M' in input:
            input_M = input['M']
            self.input_M = self.Tensor(input_M.size()).copy_(input_M)
            self.input.append(input_M)
        
        if 'Edges' in input: # new!
            input_E = input['Edges']
            self.input_E = self.Tensor(input_E.size()).copy_(input_E)
            self.input.append(input_E)

        if 'Instances' in input: # new!
            input_I = input['Instances']
            self.input_I = self.Tensor(input_I.size()).copy_(input_I)
            self.input.append(input_I)        
        
        if 'factor' in input:    
            self.strech = input['factor']

    """  Generator forward  """
    def forward(self):
        self.real_A = Variable(self.input_A)
        
        if "Ortho" in self.idict:
            self.real_O = Variable(self.input_O)
        
        if "Edges" in self.idict: # new!!
            self.real_E = Variable(self.input_E)
        
        if "Instances" in self.idict: # new!!
            self.real_I = Variable(self.input_I)        
                
        for item in range(len(self.input)):
            self.input[item] = Variable(self.input[item]).cuda(self.gpu_ids[0])
                   
        self.fake_B = self.netG.forward(self.real_A, self.real_O)
        # new!!
        if self.opt.task == 'dsm':
            self.fake_B = self.netG.forward(*self.input)
        elif self.opt.task == 'dsm_edges':
            #pdb.set_trace()
            #self.fake_B, self.pred_E = self.netG.forward(*self.input)
            self.fake_B = self.netG.forward(*self.input) 

        elif self.opt.task == 'dsm_edges_polygons':
            raise NotImplementedError('This task is not ready!')
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        
        with torch.no_grad():
            self.input = []
            self.idict = input
            
            AtoB = self.opt.which_direction == 'AtoB'
            input_A = input['A' if AtoB else 'B']
            input_B = input['B' if AtoB else 'A']
            self.input_A = self.Tensor(input_A.size()).copy_(input_A)
            self.input_B = self.Tensor(input_B.size()).copy_(input_B)
    
            self.input.append(input_A)
            
            if "Ortho" in input:
                input_O = input['Ortho']
                self.input_O = self.Tensor(input_O.size()).copy_(input_O)
                self.input.append(input_O)
                
            if 'M' in input:
                input_M = input['M']
                self.input_M = self.Tensor(input_M.size()).copy_(input_M)
                self.input.append(input_M)
            
            if 'factor' in input:    
                self.strech = input['factor']
                
    """  Discriminator backward  """
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        #pdb.set_trace()
        fake_AO = torch.cat((self.real_A, self.real_O), 1)
        fake_AB = torch.cat((fake_AO, self.real_B), 1)
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        #pdb.set_trace()
        # Real
        real_AO = torch.cat((self.real_A, self.real_O), 1)
        real_AB = torch.cat((real_AO, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

####################################################################### stop here #################################################################################################
    """  Generator backward  """
    def backward_G(self):

        """
        When we train the generator we want it to trick the discriminator. This means that we want the output of D to be close to 1,
        meaning it thinks its real. Keep that in mind. When we train G, we make the fake labels equal 1 so the optimizer tried to
        make the generator make an image that tricks D.
        """
        task_losses = {}
        
        # First, G(A) should fake the discriminator
        fake_AO = torch.cat((self.real_A, self.real_O), 1)
        fake_AB = torch.cat((fake_AO, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        
        #pdb.set_trace()

        if self.opt.loss_weights:           
            
            #Even though the images are fake, we want the discriminator to think they are real (True)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            task_losses["GAN"] = self.loss_G_GAN

            # Second, G(A) = B

            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) #Original
            
            task_losses["L1"] = self.loss_G_L1

            if self.opt.lambda_SN !=0.0:
                self.aux_loss = self.criterionSN(self.fake_B, self.real_B, self.input_M, self.strech)
                #self.aux_loss = self.criterionSN(self.fake_B, self.real_B)
                task_losses["SN"] = self.aux_loss
            ## new!
            if self.opt.task == 'dsm_edges':
                #pdb.set_trace()
                # new!!
                target_E = self.input_E.new_zeros(self.input_E[:,0,:,:].shape)
                target_E[self.input_E[:,2,:,:]>0] = 2 # background
                target_E[self.input_E[:,0,:,:]>0] = 0 # corner -> edge
                target_E[self.input_E[:,1,:,:]>0] = 1 # edge
                #self.loss_E1 = self.criterionEdges(self.side1,target_E.long())
                #self.loss_E2 = self.criterionEdges(self.side2,target_E.long())
                #self.loss_E3 = self.criterionEdges(self.side3,target_E.long())
                #self.loss_E = self.criterionEdges(self.fuse,target_E.long())/2 + (self.loss_E1 + self.loss_E2 + self.loss_E3)/6
                self.loss_E = self.criterionEdges(self.pred_E,target_E.long())
                task_losses["LE"] = self.loss_E

            elif self.opt.task == 'dsm_edges_polygons':
                raise NotImplementedError('This task is not ready!')
                
            # compute multi-task loss
            self.loss_G = 0
            
            #pdb.set_trace()
            for loss_fn, loss in task_losses.items():
                # kendall weighting
                s = self.task_weights[loss_fn]            # s := log(sigma^2)
                r = s * 0.5  
                # regularization term
                if loss_fn in ["SN", "L1"]:
                    w = 0.5 * torch.exp(-s)          # weighting (regr.)
                    self.loss_G += loss * w + r
                    
                elif loss_fn in ["GAN"]:
                    #w = torch.exp(-s)                # weighting (class.)
                
                    ## Add GAN loss
                    self.loss_G += loss * self.task_weights["GAN"]
                ## new!!
                elif loss_fn in ["LE"]:
                    w = torch.exp(-s)
                    self.loss_G += loss * w + r*2   
                     
                else:
                    logger.error("Weighting function for %s not implemented!",
                                 type(task_losses[loss_fn]))
                    raise NotImplementedError
                         
        else:  
            #pdb.set_trace()
            self.loss_G = 0
            
            #Even though the images are fake, we want the discriminator to think they are real (True)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
    
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)* self.opt.lambda_A #Original
            
            self.loss_G_SN = 0
            if self.opt.lambda_SN !=0.0: 
                #Third, Surface Normal
                self.aux_loss = self.criterionSN(self.fake_B, self.real_B, self.input_M, self.strech)*self.opt.lambda_SN
                self.loss_G_SN += self.aux_loss 
            ## new!
            self.loss_G_E = 0
            if self.opt.task == 'dsm_edges':
                self.loss_E = self.criterionEdges(self.pred_E,self.input_E)
                self.loss_G_E += self.loss_E
            
            self.loss_G = self.loss_G_L1 + self.loss_G_SN + self.loss_G_E #Original

        self.loss_G.backward()

    """  Optimization: gradient descent  """
    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    """  store losses  """
    def get_current_errors(self):
        
#        d = OrderedDict([('G_L1', self.loss_G_L1.item())])
        d = OrderedDict([('G_L1', self.loss_G_L1.item()),
                         ('G_GAN', self.loss_G_GAN.item()),
                         ('loss_D', self.loss_D.item())])
        
        if self.opt.lambda_SN !=0.0: 
            loss = self.aux_loss.item()
            name = 'loss_SN'
            d.update({name:loss})
        
        ## new!
        if self.opt.task == 'dsm_edges':
            loss_E = self.loss_E.item()
            name = 'loss_E'
            d.update({name:loss_E})

                    
        return d

    """  store images  """
    def get_current_visuals(self):
        #pdb.set_trace()
        d = OrderedDict()
        
        d['real_A'] = util.tensor2im(self.real_A.data)

        if "Ortho" in self.idict:
            d['real_O'] = util.tensor2im(self.real_O.data, spectral = True)
        if "Edges" in self.idict: # new!!
            d['real_E'] = util.tensor2im(self.real_E.data, spectral = True)
#        if "Instances" in self.idict: # new!!
#            d['real_I'] = util.tensor2im(self.real_I.data, spectral = True)        
        d['fake_B'] = util.tensor2im(self.fake_B.data)
        d['real_B'] = util.tensor2im(self.real_B.data)
        # new!!
        #pdb.set_trace()
        if self.opt.task == 'dsm_edges':
            '''
            self.vis_E = (torch.nn.Softmax(dim=1)(self.side1)+torch.nn.Softmax(dim=1)(self.side2)+torch.nn.Softmax(dim=1)(self.side3))/6+torch.nn.Softmax(dim=1)(self.fuse)/2
            #self.pred_E = torch.nn.Softmax(dim=1)(self.pred_E)
            self.vis_C = torch.nn.Softmax(dim=1)(self.pred_C)
            vis_EC = torch.stack((self.vis_C[:,1,:,:],self.vis_E[:,1,:,:]),dim=1)
            '''
            vis_E = torch.nn.Softmax(dim=1)(self.pred_E)
            #pdb.set_trace()
            d['pred_E'] = util.tensor2im(vis_E.data, spectral = True)
            
        return d

    """  save network (weights)  """
    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if self.opt.loss_weights: 
            save_filename = '%s_net_%s.pth' % (label, "task_weights")
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(self.task_weights, save_path)

    """  update learning rate  """
    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
#        for param_group in self.optimizer_D.param_groups:
#            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    """  store weights of Generator losses  """
    def get_current_LossWeights(self):
        wd = OrderedDict([('L1', self.task_weights["L1"].item()),
                          ('SN', self.task_weights["SN"].item())
                         ])
        #if self.task_weights["LE"]:
            #wd["LE"] = self.task_weights["LE"].item()
        
        if self.task_weights["GAN"]:
            wd["GAN"] = self.task_weights["GAN"].item()

            
        return wd
