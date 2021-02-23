import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np
import pdb
import matplotlib.pyplot as plt

from itertools import product
from scipy import ndimage


import u_net_resnet_50_encoder as UResNet

import architectures.resnet as resnet 


#import sys
#sys.path.append('/usr/local/lib/python2.7/dist-packages')
#sys.path.append('/usr/lib/graphviz')
#from visualize import make_dot

import sys
#sys.path.append('/home/davy_ks/project/pytorch/GDLossGAN/')
#from torchviz import make_dot, make_dot_from_trace

sys.path.append('/home/davy_ks/project/pytorch/network_investigation')
import net_investigation as invest

from io import BytesIO
from PIL import Image

from modeling.deeplab import DeepLab
from architectures.deeplabv3_resnet import DeepLabv3_plus
from architectures.u_net_resnet50 import UNetResNet50
from architectures.R2U_Net import R2U_Net, R2AttU_Net
from architectures.Coupled_R2U_Net import Coupled_R2U_Net
import architectures.R2U_Net as r2unet
#from visualize import make_dot
###############################################################################
# Functions
###############################################################################

"""  ???  """
class mergeNet(nn.Module):
    def __init__(self, net):
        super(mergeNet, self).__init__()
        
        
        self.modelG = net
      
        #self.weight = torch.nn.Parameter(torch.ones(3).float())
        self.L1 = torch.nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True)
        self.SN = torch.nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True)
        self.GAN = torch.nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True)        


    def forward(self, input):
    
        G = self.modelG(input)
        
        return G

"""  ???  """        
class mergeGAN(nn.Module):
    def __init__(self, netG, netD):
        super(mergeGAN, self).__init__()
        
        
        self.modelG = netG
        self.modelD = netD
        
#        self.L1 = torch.nn.Parameter(torch.FloatTensor([0.25]).cuda(), requires_grad=True)
#        self.SN = torch.nn.Parameter(torch.FloatTensor([0.25]).cuda(), requires_grad=True)
#        self.GAN = torch.nn.Parameter(torch.FloatTensor([0.25]).cuda(), requires_grad=True) 

        self.L1 = torch.nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True)
        self.SN = torch.nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True)
        self.GAN = torch.nn.Parameter(torch.FloatTensor([0.25]), requires_grad=True) 

        self.modelG.register_parameter("L1", self.L1)
        self.modelG.register_parameter("SN", self.SN)
        self.modelG.register_parameter("GAN", self.GAN)
                                     
    def forward(self, input):
        
        G = self.modelG(input)
        
        D = self.modelD # ???
        
        return G, D

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

"""  function: initialize weights  """    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

"""  function: define normalization layers  """
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False) # usually for GAN IN works better (yi)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
       
"""  ???  """    
class concatNet(nn.Module):
    def __init__(self, net1, net2, input_nc, output_nc, norm_layer=nn.BatchNorm2d, output_func="tanh"):
        super(concatNet, self).__init__()
        
        self.output_func = output_func
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
                        
        self.branches = nn.ModuleList([net1,net2])
        
        self.fuse_conv = nn.Sequential( 
                                nn.ReLU(True),
                                nn.ConvTranspose2d(input_nc * 256, output_nc*64,
                                                            kernel_size=4, stride=2,
                                                            padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(input_nc*64, output_nc, kernel_size=1,
                                          stride=1, padding=0, bias=use_bias),
                                nn.Tanh()
                                )
                                             
    def forward(self, inputA, inputO):

        fusion = torch.cat((self.branches[0](inputA),self.branches[1](inputO)), 1)               
        out = self.fuse_conv(fusion)  
        return out

"""  Generator  """
def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], pretrained = False, output_func="tanh", fusion = False,task='dsm'):
    print input_nc, output_nc, ngf
    netG = None
    print gpu_ids
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    if task == 'dsm_edges': ## new!!
        netG = resnet.CoupledUNetupperResnet_4(backbone = "resnet50")
    elif task == 'dsm_edges_polygons':
        netG = resnet.CoupledUNetupperResnet(backbone = "resnet50")
    elif task == 'dsm':
        
        #pdb.set_trace()
        if which_model_netG == 'unet_128':
            netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, output_func=output_func)
        elif which_model_netG == 'unet_256':        
            netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, output_func=output_func)
        elif which_model_netG == 'Coupled_unet256':
            netG = CoupledGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, output_func=output_func) 
            
        elif which_model_netG == 'DeepLab':
            netG = DeepLab(num_classes=output_nc, backbone="xception")
        elif which_model_netG == 'DeepLabv3_plus':
            netG = DeepLabv3_plus(input_nc, output_nc, os=16, pretrained=True, _print=True)
    
        elif which_model_netG == 'Single_UResNet':
                netG = UResNet.UNetWithResnet50Encoder()      
        elif which_model_netG == 'Coupled_UResNet':
                netG = UResNet.CoupledUNetResnet() # my written 2-nd version, where the number of channels in decoder is smaller
        elif which_model_netG == 'Coupled_UResNet18':
            #netG = resnet.CoupledUNetlowerResnet(backbone = "resnet18")
            netG = UResNet.CoupledUNetLowerResnet(backbone = "resnet18") # my written 2-nd version, where the number of channels in decoder is smaller
        elif which_model_netG == 'Coupled_UResNet34':
            #netG = resnet.CoupledUNetlowerResnet(backbone = "resnet34")
            netG = UResNet.CoupledUNetLowerResnet(backbone = "resnet34") # my written 2-nd version, where the number of channels in decoder is smaller
        elif which_model_netG == 'Coupled_UResNet50':
            netG = resnet.CoupledUNetupperResnet_4(backbone = "resnet50") ## currently using! change generator architecture here!!!! (new!)
        elif which_model_netG == 'Coupled_UResNet101':
            netG = resnet.CoupledUNetupperResnet(backbone = "resnet101")
        elif which_model_netG == 'Coupled_UResNet152':
            netG = resnet.CoupledUNetupperResnet(backbone = "resnet152")
    
        elif which_model_netG == 'Single_R2U_Net': 
                netG = R2U_Net()        
        elif which_model_netG == 'Coupled_R2U_Net': 
                netG = Coupled_R2U_Net()
                r2unet.init_weights(netG)
    
    
        elif which_model_netG == 'R2AttU_Net': 
                netG = R2AttU_Net()
                r2unet.init_weights(netG) 
    
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    else:
        raise NameError('task not defined, try "dsm", "dsm_edges" or "dsm_edges_polygons"')
#    pdb.set_trace()    
#    if which_model_netG not in ["Coupled_UResNet", "DeepLab", "DeepLabv3_plus",'Corentin_UNetResNet50']:
#        netG.apply(weights_init)
    
#    if len(gpu_ids) > 0:
#        pdb.set_trace()
#        netG.cuda(device_id=gpu_ids[0])
        
        
        
#    netG.apply(weights_init)
#    for param in netG.parameters(): 
#        print param.data   
#        pdb.set_trace() 
            
    return netG

"""  Discriminator  """
def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    #pdb.set_trace()
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids) ## currently using! change architectures here! (new!)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        #netD.cuda(device_id=gpu_ids[0])
        netD.cuda(gpu_ids[0])
        
    netD.apply(weights_init) ### ????
    return netD

"""  function: print numbers of network parameters  """
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

"""  GAN loss  """
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
            print "loss = nn.MSELoss()"
        else:
            self.loss = nn.BCELoss()
            print "loss = nn.BCELoss()"

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor) ### ??????

"""  surface normal loss  """  ## to be learned
class BuidingSurfaceNormalLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(BuidingSurfaceNormalLoss, self).__init__()
        self.loss = None

    def __call__(self, fake_B, real_B, mask, stretch):
    
        # deNormalization of real_B and fake_B from [-1,1] to standart normal values        
        fake =(((fake_B + 1)*(0.5*(stretch[0][1].float()-stretch[0][0])).cuda().view(-1,1,1,1))+stretch[0][0].cuda().view(-1,1,1,1)) ### stretch??
        real =(((real_B + 1)*(0.5*(stretch[0][1].float()-stretch[0][0])).cuda().view(-1,1,1,1))+stretch[0][0].cuda().view(-1,1,1,1))

        # Two 3x3 Filter in x/y-direction (with padding on border):
                
        #dx
#        axis_1 = torch.tensor([[-1.0, 0.0, 1.0],
#                             [-1.0, 0.0, 1.0],
#                             [-1.0, 0.0, 1.0]]).unsqueeze(0).unsqueeze(0)

        axis_1 = torch.tensor([[1.0, 0.0, -1.0],
                             [1.0, 0.0, -1.0],
                             [1.0, 0.0, -1.0]]).unsqueeze(0).unsqueeze(0)
#        #dy                        
#        axis_0 = torch.tensor([[-1.0, -1.0, -1.0],
#                             [0.0, 0.0, 0.0],
#                             [1.0, 1.0, 1.0]]).unsqueeze(0).unsqueeze(0)

        #dy                        
        axis_0 = torch.tensor([[1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0],
                             [-1.0, -1.0, -1.0]]).unsqueeze(0).unsqueeze(0)
                                                          
        if fake.is_cuda:
            axis_1 = axis_1.cuda()
            axis_0 = axis_0.cuda()
        
        # Define convolution operations for x/y-directions 
        # Padding is set to 0 as we add the special borders to the image directly            
        conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        conv_x.weight = nn.Parameter(axis_1, requires_grad=False)
        
        conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        conv_y.weight = nn.Parameter(axis_0, requires_grad=False)
        
        # Set the replication padding at image borders
        m = nn.ReplicationPad2d(1)
        
        # Compute normals for fake and real image
        dx_fake = conv_x(m(fake))/(6*0.5*1.0)
        dy_fake = conv_y(m(fake))/(6*0.5*1.0)
        
        dx_real = conv_x(m(real))/(6*0.5*1.0)
        dy_real = conv_y(m(real))/(6*0.5*1.0)
        
        # Define the tensor ones of the same size as for dx and dy dirations
        # This additional tensor is for z direction
        ones = torch.ones([fake.size()[0],1,fake.size()[2],fake.size()[3]])
        #pdb.set_trace()
        if fake.is_cuda:
            ones=ones.cuda()
        
        #normals_fake = torch.cat((dy_fake,dx_fake,ones),dim=1)
        #normals_real = torch.cat((dy_real,dx_real,ones),dim=1)

        #plt.show(plt.imshow(fake.data.cpu().numpy()[0,0,:,:]))
        #plt.show(plt.imshow(normals.data.cpu().numpy()[0,:,:,:].transpose(1,2,0)))
        
        # Define eight neighbourhood for each central pixels
        # The shape of kernel is [8, 1, 3, 3]
        neighbourhood_kernel = torch.tensor([[[1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                                             [[0.0, 1.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                                             [[0.0, 0.0, 1.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                                             [[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                                             [[0.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 0.0, 0.0]],
                                             [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[1.0, 0.0, 0.0]],
                                             [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 1.0, 0.0]],
                                             [[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 1.0]]]).unsqueeze(1)        
        if fake.is_cuda:
            neighbourhood_kernel = neighbourhood_kernel.cuda()
        
        # Define convolution operation which applys neighbourhood_kernel on fake and real images
        # It produces the 4D tensor of shape [3, 8, 256, 256]
        # Along dim=1 all eight neighbouring pixels are stored    
        conv_neighb = nn.Conv2d(1, neighbourhood_kernel.size()[1], kernel_size=3, stride=1, padding=0, bias=False)
        conv_neighb.weight = nn.Parameter(neighbourhood_kernel, requires_grad=False)
        
        # The dx and dy components of normal vectors for fake and real images
        # Along dim=1 all eight neighbouring dx and dy components are stored
        neighbours_dy_fake = conv_neighb(m(dy_fake))
        neighbours_dx_fake = conv_neighb(m(dx_fake))
        
        neighbours_dy_real = conv_neighb(m(dy_real))
        neighbours_dx_real = conv_neighb(m(dx_real))

        # Compute the product of central normal to its eight neighbouring normals
        # vector_n1 x vector_ni where i in range [0,8] 
        # The product is computed separate for x and y normal directions
        prod_dy_fake = torch.mul(neighbours_dy_fake,dy_fake)
        prod_dx_fake = torch.mul(neighbours_dx_fake,dx_fake)
        
        prod_dy_real = torch.mul(neighbours_dy_real,dy_real)
        prod_dx_real = torch.mul(neighbours_dx_real,dx_real)
        #pdb.set_trace()

        # Compute the absolute value/magnitude of each normal
        magnitude_nghbfake = torch.sqrt(torch.add(neighbours_dy_fake**2,neighbours_dx_fake**2)+1)
        magnitude_fake = torch.sqrt(torch.add(dy_fake**2,dx_fake**2)+1)

        magnitude_nghbreal = torch.sqrt(torch.add(neighbours_dy_real**2,neighbours_dx_real**2)+1)
        magnitude_real = torch.sqrt(torch.add(dy_real**2,dx_real**2)+1)        
        
        # Scalar Product / Dot Product
        # Find the cos angle between two normals within 8 neighbourhood for fake and real images
        cosangle_fake = torch.div(torch.add(torch.add(prod_dy_fake,prod_dx_fake),1),torch.mul(magnitude_nghbfake,magnitude_fake))        
        cosangle_real = torch.div(torch.add(torch.add(prod_dy_real,prod_dx_real),1),torch.mul(magnitude_nghbreal,magnitude_real))
        
        # As we want to concentrate only on the areas where the buildings are situated 
        # we filter the rest of the area by setting it to zeros
        # We make it only for the real image, as the fake image need to be filtered later
        # because of additional condition
        #tofilter_cosangle_fake = torch.mul(cosangle_fake, mask)        
        tofilter_cosangle_real = torch.mul(cosangle_real, mask)
        
        # As our goal is to make the neighbour normals be parallel and look at the same direction 
        # we select only those 9 neighbour normals which belong to the same plane and, as a result,
        # look at the same diraction. We make this conditioning on real image as on fake image 
        # at the beginning all normals randomly distributed
        # validmask is the mask which contains 0 and 1 values
        validmask=(torch.sum(tofilter_cosangle_real>=np.cos(np.deg2rad(30)),dim=1)/8).unsqueeze(1)

        
        # We finally filter the cos angles of fake image with the rest of valid normals regarding our conditions
        # And sumed up the values along dim = 1, within 256x256 and along batches
        #tofilter_cosangle_fake = torch.sum(torch.mul(cosangle_fake, validmask.type(torch.cuda.FloatTensor)),dim=1)/8
        tofilter_cosangle_fake = torch.sum(torch.mul(cosangle_fake, validmask.type(torch.cuda.FloatTensor)))       
        
        #pdb.set_trace() 
        # 8 comes from 8 channes along dim =1 [batch, 8, height, width]
        # len(torch.nonzero(validmask)) provides the number of valid element along the whole batch
        # As we aim to minimize the loss function, we subtract cos angles from 1
        if  len(torch.nonzero(validmask))== 0:
            denominator = 1e-12
        else:
            denominator = len(torch.nonzero(validmask))
        #return 1 - tofilter_cosangle_fake/(len(torch.nonzero(validmask))*8)
        return 1 - tofilter_cosangle_fake/(denominator*8)

"""  normal loss???  """        
class NormalLoss(nn.Module):        
    def __init__(self, tensor=torch.FloatTensor):
        super(NormalLoss, self).__init__()
        self.loss = None
        
    def __call__(self, fakeIMG, realIMG, mask, stretch):
        
        mask[mask==255]=1
        #pdb.set_trace()       
        fake =((torch.add(fakeIMG,1)*(0.5*(stretch[0][1].float()-stretch[0][0].float())).cuda().view(-1,1,1,1))+stretch[0][0].float().cuda().view(-1,1,1,1))
        real =((torch.add(realIMG,1)*(0.5*(stretch[0][1].float()-stretch[0][0].float())).cuda().view(-1,1,1,1))+stretch[0][0].float().cuda().view(-1,1,1,1))
      
        loss = 0.0
        not_empty_batches = 0

        # Two 3x3 Filter in x/y-direction (with padding on border):
                
        #dx
#        axis_1 = torch.tensor([[-1.0, 0.0, 1.0],
#                             [-1.0, 0.0, 1.0],
#                             [-1.0, 0.0, 1.0]]).unsqueeze(0).unsqueeze(0)

        axis_1 = torch.tensor([[1.0, 0.0, -1.0],
                             [1.0, 0.0, -1.0],
                             [1.0, 0.0, -1.0]]).unsqueeze(0).unsqueeze(0)
#        #dy                        
#        axis_0 = torch.tensor([[-1.0, -1.0, -1.0],
#                             [0.0, 0.0, 0.0],
#                             [1.0, 1.0, 1.0]]).unsqueeze(0).unsqueeze(0)

        #dy                        
        axis_0 = torch.tensor([[1.0, 1.0, 1.0],
                             [0.0, 0.0, 0.0],
                             [-1.0, -1.0, -1.0]]).unsqueeze(0).unsqueeze(0)
                             
        if fake.is_cuda:
            axis_1 = axis_1.cuda()
            axis_0 = axis_0.cuda()
                    
        conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        conv_x.weight = nn.Parameter(axis_1, requires_grad=False)
        
        conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        conv_y.weight = nn.Parameter(axis_0, requires_grad=False)
        
        m = nn.ReplicationPad2d(1)
        
        dx_fake = conv_x(m(fake))/(6*0.5*1.0)
        dy_fake = conv_y(m(fake))/(6*0.5*1.0)
        
        dx_real = conv_x(m(real))/(6*0.5*1.0)
        dy_real = conv_y(m(real))/(6*0.5*1.0)
                
#        dx_fake = (F.conv2d(m(fake), Variable(axis_1, requires_grad=False)))/(6*0.5*1.0) #(6*0.5*1.0) #(6*0.01*0.02) 
#        dy_fake = (F.conv2d(m(fake), Variable(axis_0, requires_grad=False)))/(6*0.5*1.0) #(6*0.5*1.0) #(6*0.01*0.02) 
#
#        dx_real = (F.conv2d(m(real), Variable(axis_1, requires_grad=False)))/(6*0.5*1.0) #(6*0.5*1.0) #(6*0.01*0.02) 
#        dy_real = (F.conv2d(m(real), Variable(axis_0, requires_grad=False)))/(6*0.5*1.0) #(6*0.5*1.0) #(6*0.01*0.02)        
        
        
        ones = torch.ones([fake.size()[0],1,fake.size()[2],fake.size()[3]])
        
        if fake.is_cuda:
            ones=ones.cuda()
            
        

        N,C,_,_ = fake.size()
        
        
        normals_fake=torch.cat((dy_fake.view(N,C,-1), dx_fake.view(N,C,-1), ones.view(N,C,-1)), dim=1)
        normals_real=torch.cat((dy_real.view(N,C,-1), dx_real.view(N,C,-1), ones.view(N,C,-1)), dim=1)
        
        #plt.show(plt.imshow(normals_real.view(N,3,256,256).data.cpu().numpy()[0,:,:,:].transpose(1,2,0)))
        #scipy.misc.toimage(self.real_B[0,0,:,:]).save('self.real_B.jpg')
        
        prod = torch.sum((torch.mul( normals_fake, normals_real).squeeze(-1).squeeze(-1)),dim=1).unsqueeze(1) #x1*x2+y1*y2+z1*z2
        
        magnitude_fake = torch.sqrt(torch.sum( normals_fake**2, dim=1 ))
        magnitude_real = torch.sqrt(torch.sum( normals_real**2, dim=1 ))
        
        cosangle = torch.div(prod,(magnitude_fake*magnitude_real).unsqueeze(1))
        
        filter_cosangle = torch.mul(cosangle, mask.view(N,C,-1))
        
        valid = torch.nonzero(mask)
                
        #cosangle=prod/(magnitude_fake*magnitude_real)        
        #pdb.set_trace()
        
        #mask = mask.type(torch.cuda.ByteTensor)
        #masked_angles = torch.cat((torch.masked_select(cosangle[:,0,:].unsqueeze(1), mask.view(N,C,-1))[None,:],torch.masked_select(cosangle[:,1,:].unsqueeze(1), mask.view(N,C,-1))[None,:],torch.masked_select(cosangle[:,2,:].unsqueeze(1), mask.view(N,C,-1))[None,:]))
        
        
        #if (1 - torch.div(torch.sum(torch.div(torch.sum((tofilter_cosangle_real),dim=1),3)),denominator))<0.0:      
        #pdb.set_trace()

        if  len(valid)== 0:
            denominator = 1e-12
        else:
            denominator = len(valid)
        
        #plt.show(plt.imshow(torch.sum((filter_cosangle),dim=1).unsqueeze(1).view(5,1,256,256).data.cpu().numpy()[1,0,:,:]))
        #plt.show(plt.imshow(fake.data.cpu().numpy()[0,0,:,:]))
            
        return 1 - torch.div(torch.sum(filter_cosangle),denominator)                
        #return 1 - torch.mean(torch.div(prod,(magnitude_fake*magnitude_real).unsqueeze(1)))
        #return 1 - torch.mean( masked_angles) 

"""  unet blocks???  """                
def net_blocks(ngf, num_downs, norm_layer, use_dropout):
    unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
    
    for i in range(num_downs - 5):
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
    unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
    unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
    
    return unet_block

"""  ???  """        
class WnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], output_func="tanh"):
        super(WnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        
        net = net_blocks(ngf, num_downs, norm_layer, use_dropout)
        net = InputBlock(input_nc, ngf, net, outermost=True, norm_layer=norm_layer, outer_nc_out=output_nc)
             
        self.model = net

    def forward(self, input):
        
        #fusion = torch.cat((self.branches[0](inputA),self.branches[1](inputO)), 1)
        #out = self.net1(inputA)
        

        ## Network visualization         
        #make_dot(self.model(inputA)).view()
        #print "WnetGenerator output size:", self.model(input).size()
        
        return self.model(input)

"""  ???  """
class InputBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outer_nc_out=None):
        super(InputBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if outer_nc_out is None:
            outer_nc_out = outer_nc

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)

        model = [downconv] + [submodule]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

"""  ???  """
class OutBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outer_nc_out=None,
                 output_func="tanh"):
        super(OutBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if outer_nc_out is None:
            outer_nc_out = outer_nc

        uprelu = nn.ReLU(True)

        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc_out,
                                    kernel_size=4, stride=2,
                                    padding=1)
        if output_func == "tanh":
            up = [uprelu, upconv, nn.Tanh()]
        elif output_func == "none":
            up = [uprelu, upconv]
        else:
            raise ValueError("Invalid output function: %s"%output_func)
        
        model = [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)

"""  ???  """            
class CoupledGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], output_func="tanh"):
        super(CoupledGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        

        unet_block = CoNetSkipConnectionBlock(output_nc, ngf, norm_layer=norm_layer, innermost=True)

        self.model = unet_block

    def forward(self, *inputs):
        #pdb.set_trace()
        #invest.network_investigation(self.model, input[0], UnetSkipConnectionBlock)
        if self.gpu_ids and isinstance(inputs[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, inputs, self.gpu_ids)
        else:
            #make_dot(self.model(input)).view()
            return self.model(*inputs)

    def freeze_bn(self):
        #pdb.set_trace()
        for m in self.modules():
            print m
            #if isinstance(m, nn.InstanceNorm2d): 
            if isinstance(m, nn.BatchNorm2d):
                #pdb.set_trace()
                m.eval() 
                    
"""  ???  """
class CoNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outer_nc_out=None,
                 output_func="tanh"):
        super(CoNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if outer_nc_out is None:
            outer_nc_out = outer_nc

        def down_block(in_filters, out_filters, n_layer=True):
            
            if n_layer == True:
                block = nn.Sequential(nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(in_filters, out_filters, kernel_size=4,stride=2, padding=1, bias=use_bias),
                                    norm_layer(out_filters)) 
            else:
                block = nn.Sequential(nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(in_filters, out_filters, kernel_size=4,stride=2, padding=1, bias=use_bias))                                      
            return block

        def up_block(in_filters, out_filters, dropout=True):
            
            if dropout == True:
                block = nn.Sequential(nn.ReLU(True),
                                    nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    norm_layer(out_filters),
                                    nn.Dropout(0.5))
            else:
                block = nn.Sequential(nn.ReLU(True),
                                    nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    norm_layer(out_filters))    
            return block 
            
        # DSM Branch
        self.downoutmost = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
                                     
        
        self.downconv1 = down_block(inner_nc, inner_nc*2)  

        self.downconv2 = down_block(inner_nc*2, inner_nc*4)

        self.downconv3 = down_block(inner_nc*4, inner_nc*8)

        self.downconv4 = down_block(inner_nc*8, inner_nc*8)

        self.downconv5 = down_block(inner_nc*8, inner_nc*8)
                                     
        self.downconv6 = down_block(inner_nc*8, inner_nc*8)
                                     
        self.downconv7 = down_block(inner_nc*8, inner_nc*8, n_layer=False)        
        
        
        
        # Ortho Branch
        self.ortho_downoutmost = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
                                     
        
        self.ortho_downconv1 = down_block(inner_nc, inner_nc*2)  

        self.ortho_downconv2 = down_block(inner_nc*2, inner_nc*4)

        self.ortho_downconv3 = down_block(inner_nc*4, inner_nc*8)

        self.ortho_downconv4 = down_block(inner_nc*8, inner_nc*8)

        self.ortho_downconv5 = down_block(inner_nc*8, inner_nc*8)
                                     
        self.ortho_downconv6 = down_block(inner_nc*8, inner_nc*8)
                                     
        self.ortho_downconv7 = down_block(inner_nc*8, inner_nc*8, n_layer=False)  
                                     
                                                                                                                       
        self.up = up_block(inner_nc * 16, inner_nc * 8, dropout=False)

        self.upconv6 = up_block(inner_nc * 24, inner_nc * 8)
                                    
        self.upconv5 = up_block(inner_nc * 24, inner_nc * 8)

        self.upconv4 = up_block(inner_nc * 24, inner_nc * 8)
                                                                                                            
        self.upconv3 = up_block(inner_nc * 24, inner_nc * 4, dropout=False)

        self.upconv2 = up_block(inner_nc * 12, inner_nc * 2, dropout=False)                                                                                                                            

        self.upconv1 = up_block(inner_nc * 6, inner_nc, dropout=False) 
                                    
        if output_func == "tanh":
            self.last_up = nn.Sequential(nn.ReLU(True),                                    
                                        nn.ConvTranspose2d(inner_nc * 3, outer_nc, kernel_size=4, stride=2, padding=1),
                                        nn.Tanh()) 
        elif output_func == "none":
            self.last_up = nn.Sequential(nn.ReLU(True),
                                        nn.ConvTranspose2d(inner_nc * 3, outer_nc, kernel_size=4, stride=2, padding=1), # in corrected no bias
                                        )  
                

    def forward(self, x1, x2):
        #pdb.set_trace()
        
        
        outermost_1 = self.downoutmost(x1)
        outermost_2 = self.ortho_downoutmost(x2)
        
        #Encoder 1
        layer11 = self.downconv1(outermost_1)        
        layer21 = self.downconv2(layer11)              
        layer31 = self.downconv3(layer21)                       
        layer41 = self.downconv4(layer31)        
        layer51 = self.downconv5(layer41)                                
        layer61 = self.downconv6(layer51)        
        layer71 = self.downconv7(layer61)        
        
        # Encoder 2
        layer12 = self.ortho_downconv1(outermost_2)
        layer22 = self.ortho_downconv2(layer12)        
        layer32 = self.ortho_downconv3(layer22)        
        layer42 = self.ortho_downconv4(layer32)
        layer52 = self.ortho_downconv5(layer42)
        layer62 = self.ortho_downconv6(layer52)
        layer72 = self.ortho_downconv7(layer62)        
        
        
        # Common Decoder
        fusion = torch.cat((layer71,layer72), 1)
        up7 = torch.cat((layer61,layer62,self.up(fusion)),1)
        up6 = torch.cat((layer51,layer52,self.upconv6(up7)),1)
        up5 = torch.cat((layer41,layer42,self.upconv5(up6)),1)
        up4 = torch.cat((layer31,layer32,self.upconv4(up5)),1)
        up3 = torch.cat((layer21,layer22,self.upconv3(up4)),1)
        up2 = torch.cat((layer11,layer12,self.upconv2(up3)),1)
        up1 = torch.cat((outermost_1,outermost_2,self.upconv1(up2)),1)
        #pdb.set_trace()
        out = self.last_up(up1)
                        
        return out    

"""  ???  """
class CoNetLateShareSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outer_nc_out=None,
                 output_func="tanh"):
        super(CoNetLateShareSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        #pdb.set_trace()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if outer_nc_out is None:
            outer_nc_out = outer_nc

        def down_block(in_filters, out_filters, n_layer=True):
            
            if n_layer == True:
                block = nn.Sequential(nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(in_filters, out_filters, kernel_size=4,stride=2, padding=1, bias=use_bias),
                                    norm_layer(out_filters)) 
            else:
                block = nn.Sequential(nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(in_filters, out_filters, kernel_size=4,stride=2, padding=1, bias=use_bias))                                      
            return block

        def up_block(in_filters, out_filters, dropout=True):
            
            if dropout == True:
                block = nn.Sequential(nn.ReLU(True),
                                    nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    norm_layer(out_filters),
                                    nn.Dropout(0.5))
            else:
                block = nn.Sequential(nn.ReLU(True),
                                    nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    norm_layer(out_filters))    
            return block            
    

        # DSM Branch
        self.downoutmost = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
                                     
        
        self.downconv1 = down_block(inner_nc, inner_nc*2)  

        self.downconv2 = down_block(inner_nc*2, inner_nc*4)

        self.downconv3 = down_block(inner_nc*4, inner_nc*8)

        self.downconv4 = down_block(inner_nc*8, inner_nc*8)
        
        

        # Ortho Image Branch
        self.ortho_downoutmost = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
                                                         
        self.ortho_downconv1 = down_block(inner_nc, inner_nc*2)
                                     
        self.ortho_downconv2 = down_block(inner_nc*2, inner_nc*4)

        self.ortho_downconv3 = down_block(inner_nc*4, inner_nc*8)

        self.ortho_downconv4 = down_block(inner_nc*8, inner_nc*8)
                                                                                                                       

        ## From down 5 layer share the weights between DSM and Ortho Branches                                                                                                              
        self.downconv5 = down_block(inner_nc*8, inner_nc*8)
                                     
        self.downconv6 = down_block(inner_nc*8, inner_nc*8)
                                     
        self.downconv7 = down_block(inner_nc*8, inner_nc*8, n_layer=False)
        
                                     
                                                                                                                       
        self.up = up_block(inner_nc * 16, inner_nc * 8, dropout=False)

        self.upconv6 = up_block(inner_nc * 24, inner_nc * 8)
                                    
        self.upconv5 = up_block(inner_nc * 24, inner_nc * 8)

        self.upconv4 = up_block(inner_nc * 24, inner_nc * 8)
                                                                                                            
        self.upconv3 = up_block(inner_nc * 24, inner_nc * 4, dropout=False)

        self.upconv2 = up_block(inner_nc * 12, inner_nc * 2, dropout=False)                                                                                                                            

        self.upconv1 = up_block(inner_nc * 6, inner_nc, dropout=False) 
                                    
        if output_func == "tanh":
            self.last_up = nn.Sequential(nn.ReLU(True),                                    
                                        nn.ConvTranspose2d(inner_nc * 3, outer_nc, kernel_size=4, stride=2, padding=1),
                                        nn.Tanh()) 
        elif output_func == "none":
            self.last_up = nn.Sequential(nn.ReLU(True),
                                        nn.ConvTranspose2d(inner_nc * 3, outer_nc, kernel_size=4, stride=2, padding=1), # in corrected no bias
                                        ) 
                

    def forward(self, x1, x2):
        
        #pdb.set_trace()
        
        
        #Encoder 1
        outermost_1 = self.downoutmost(x1) 
        layer11 = self.downconv1(outermost_1)        
        layer21 = self.downconv2(layer11)              
        layer31 = self.downconv3(layer21)                       
        layer41 = self.downconv4(layer31) 
        
        layer51 = self.downconv5(layer41) ## shared                                
        layer61 = self.downconv6(layer51) ## shared         
        layer71 = self.downconv7(layer61) ## shared         
        
        # Encoder 2
        outermost_2 = self.ortho_downoutmost(x2) 
        layer12 = self.ortho_downconv1(outermost_2) 
        layer22 = self.ortho_downconv2(layer12)        
        layer32 = self.ortho_downconv3(layer22)        
        layer42 = self.ortho_downconv4(layer32)
        
        layer52 = self.downconv5(layer42) ## shared  
        layer62 = self.downconv6(layer52) ## shared  
        layer72 = self.downconv7(layer62) ## shared         
        
        
        # Common Decoder
        fusion = torch.cat((layer71,layer72), 1)
        up7 = torch.cat((layer61,layer62,self.up(fusion)),1)
        up6 = torch.cat((layer51,layer52,self.upconv6(up7)),1)
        up5 = torch.cat((layer41,layer42,self.upconv5(up6)),1)
        up4 = torch.cat((layer31,layer32,self.upconv4(up5)),1)
        up3 = torch.cat((layer21,layer22,self.upconv3(up4)),1)
        up2 = torch.cat((layer11,layer12,self.upconv2(up3)),1)
        up1 = torch.cat((outermost_1,outermost_2,self.upconv1(up2)),1)
        #pdb.set_trace()
        out = self.last_up(up1)
                        
        return out   

"""  u-net generator???  """           
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], output_func="tanh"):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(input_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, outer_nc_out=output_nc,
                                             output_func=output_func)

        self.model = unet_block
            
    def forward(self, input):
        
        #invest.network_investigation(self.model, input[0], UnetSkipConnectionBlock)
        
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            #make_dot(self.model(input)).view()
            return self.model(input)

"""  u-net skip connection  """
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, outer_nc_out=None,
                 output_func="tanh"):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if outer_nc_out is None:
            outer_nc_out = outer_nc

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        #self.test=downconv
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc_out,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_func == "tanh":
                up = [uprelu, upconv, nn.Tanh()]
            elif output_func == "none":
                up = [uprelu, upconv]
            else:
                raise ValueError("Invalid output function: %s"%output_func)
            #up = [uprelu, upconv, nn.Tanh()]
            
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        #pdb.set_trace()
        self.model = nn.Sequential(*model)

    def forward(self, x):
        #pdb.set_trace()
        if self.outermost:
            return self.model(x)
        else:
            
            return torch.cat([x, self.model(x)], 1)

"""  Discriminator: PatchGAN  """            
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        self.test=nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        #pdb.set_trace()
        #inp=self.test(input[0])
        #pdb.set_trace()
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
