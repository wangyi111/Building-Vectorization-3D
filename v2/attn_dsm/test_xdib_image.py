import time
import os
from options.test_options import TestOptions
from data.xdibias_dataset import XdibiasDSMLoader
#from data.xdibias_dataset_PC import XdibiasPCDSMLoader
from models.models import create_model
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np
import xdibias
import pdb
import yaml 
from PIL import Image

def build_weights(tilesize, overlap):
    th,tw = tilesize
    y,x = np.mgrid[0:th,0:tw]
    y = y.astype(np.float32) / (overlap[0] - 1.0)
    y[y>=1.0] = 1.0
    x = x.astype(np.float32) / (overlap[1] - 1.0)
    x[x>=1.0] = 1.0
    
    w = np.dstack((y,x,y[::-1,:],x[:,::-1])).min(axis=2)
    return w
    

if __name__ == "__main__":
    optparse = TestOptions()
    #optparse.parser.add_argument("DSM", nargs=1, help="Input DSM image") #"DSM"
    #optparse.parser.add_argument("Ortho", nargs=1, help="Input Ortho image") #"Ortho"
    optparse.parser.add_argument("Out", nargs=1, help="Output image") #"Out"
    optparse.parser.add_argument('--overlap', type=int, default=0, help='Overlap between tiles')
    optparse.parser.add_argument('--ortho', default=None, help='optional ortho image')
    optparse.parser.add_argument('--xd_loader', type=str, default='XdibiasDSMLoader',
                        help="xdibias data loader")
    
    
    opt = optparse.parse()
    #pdb.set_trace()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.which_epoch = '100'

    with open(os.path.join(opt.dataroot, "config.yaml")) as cf:
        config = yaml.load(cf)
    
    """ Load training dataset """
    dataset = XdibiasDSMLoader(opt, config, config["data"]["roi_test"])
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size = opt.batchSize,
                                               shuffle = False,
                                               #num_workers = int(opt.nThreads)
                                               num_workers = 0
                                               )
                                         
#    data_loader = CreateDataLoader(opt)
#    #dataset = data_loader.load_data()
#    dataset = data_loader.dataset
    model = create_model(opt)
    
    
    
    print dataset.config
    print("Creating output image")
    Out = xdibias.Image(os.path.join(opt.test_outdir,opt.Out[0]), createNew=True)
    Out.Rows = dataset.DSM.Rows
    Out.Columns = dataset.DSM.Columns
    Out.Channels = dataset.DSM.Channels
    Out.BitsPerChannel = 32
    Out.copyMetadata(dataset.DSM,
                     imagetype = True,
                     geo = True,
                     radiometry = True)

    Out2 = xdibias.Image(os.path.join(opt.test_outdir,'result_edges'), createNew=True)
    Out2.Rows = dataset.Edges.Rows
    Out2.Columns = dataset.Edges.Columns
    Out2.Channels = [1,2,3]
    Out2.BitsPerChannel = 32
    Out2.copyMetadata(dataset.DSM,
                     imagetype = True,
                     geo = True,
                     radiometry = True)

    y=0
    ov2 = opt.overlap/2
    ove = opt.fineSize - ov2
    i=0
    n_cl = 3 ##new!
    # compute weight map for overlap merging
    weights = build_weights((opt.fineSize, opt.fineSize), (opt.overlap, opt.overlap)) + 1e-2
    output_line = np.zeros((opt.fineSize,Out.Columns), dtype=np.float32)
    output_weights = np.zeros((opt.fineSize,Out.Columns), dtype=np.float32)
    ##new!
    weights2 = np.zeros((opt.fineSize,opt.fineSize,n_cl), dtype=np.float32)
    for ii in range(0,n_cl):
        weights2[:,:,ii] = build_weights((opt.fineSize, opt.fineSize), (opt.overlap, opt.overlap)) + 1e-2    
    output2_line = np.zeros((opt.fineSize,Out.Columns,n_cl), dtype=np.float32)
    output2_weights = np.zeros((opt.fineSize,Out.Columns,n_cl), dtype=np.float32)

    full_line = np.zeros((opt.fineSize,Out.Columns), dtype=np.float32)
    full_weights = np.zeros((opt.fineSize,Out.Columns), dtype=np.float32)
    ##new!
    full2_line = np.zeros((opt.fineSize,Out.Columns,n_cl), dtype=np.float32)
    full2_weights = np.zeros((opt.fineSize,Out.Columns,n_cl), dtype=np.float32)

    temp = np.zeros((opt.fineSize,Out.Columns), dtype=np.float32)
    
    prev = 0
    
    flag = False
    #pdb.set_trace()
    # smooth to edges
    while i < dataset.DSM.Rows - opt.overlap:
        i = min(i, dataset.DSM.Rows - opt.fineSize)
        j=0
        while j < dataset.DSM.Columns - opt.overlap:            
            # special case for last tile            
            j = min(j, dataset.DSM.Columns - opt.fineSize)
            
            print("Processing",i,j)
             
            input = []
                       
            input_dsm, norm_params = dataset.getPatch(dataset.DSM,i,j,norm=True)
            input_dsm = dataset.transform(Image.fromarray(input_dsm))
            
            
            # create a one sample batch, as required by the network
            A = torch.stack([input_dsm])
            
            if dataset.Ortho:
                input_ortho = dataset.getPatch(dataset.Ortho,i,j,norm="intensity")
                input_ortho = dataset.transform(Image.fromarray(input_ortho))
                
                # create a one sample batch, as required by the network
                O = torch.stack([input_ortho])
                
            if dataset.MASK:
                input_mask = dataset.getPatch(dataset.MASK,i,j,norm=False)
                
                if opt.xd_loader =='XdibiasDSMLoader':
                    input_mask = dataset.transform(Image.fromarray(input_mask))
                else:
                    input_mask = dataset.transform(Image.fromarray(input_mask * 255, mode='L').convert('1'))
                
                # create a one sample batch, as required by the network
                M = torch.stack([input_mask])
            
            #real_A = Variable(A, volatile=True)
            #real_O = Variable(O, volatile=True)
            with torch.no_grad():
                real_A = Variable(A.float(),requires_grad=False)  
                
                if dataset.Ortho:
                    real_O = Variable(O.float(),requires_grad=False) 
                if dataset.MASK:   
                    real_M = Variable(M.float(),requires_grad=False) 

            if opt.gpu_ids:
                real_A = real_A.cuda()
                input.append(real_A)
                
                if dataset.Ortho:
                    real_O = real_O.cuda()
                    input.append(real_O)
                if dataset.MASK: 
                    real_M = real_M.cuda()
                    input.append(real_M)
            
            #pdb.set_trace()
            
            fwd_start = time.time()
            #fake_B = model.netG.forward(real_A,real_O)
            fake_B,pred_E = model.netG.forward(*input)
            '''
            pred_E = (torch.nn.Softmax(dim=1)(side1) + 
                      torch.nn.Softmax(dim=1)(side2) +
                      torch.nn.Softmax(dim=1)(side3) +
                      torch.nn.Softmax(dim=1)(side4) +
                      torch.nn.Softmax(dim=1)(side5) +
                      torch.nn.Softmax(dim=1)(fuse))/6
            '''
            '''
            vis_E = (torch.nn.Softmax(dim=1)(side1)+torch.nn.Softmax(dim=1)(side2)+torch.nn.Softmax(dim=1)(side3))/6+torch.nn.Softmax(dim=1)(fuse)/2
            vis_C = torch.nn.Softmax(dim=1)(pred_C)
            pred_E = torch.stack((vis_C[:,1,:,:],vis_E[:,1,:,:],vis_E[:,0,:,:]),dim=1)
            '''
            pred_E = torch.nn.Softmax(dim=1)(pred_E)
            #fake_B = real_A
            t_fwd = time.time() - fwd_start

            
	       # Network with learned parameters                   
            #fake_B = model.concat_net.modelG.forward(real_A,real_O)

            if isinstance(fake_B.data, torch.cuda.FloatTensor):
                fake_B = fake_B.cpu()
            if isinstance(pred_E.data, torch.cuda.FloatTensor):
                pred_E = pred_E.cpu()
                                         
            Out_tile = dataset.normalize(fake_B.data.numpy()[0,0], 
                                         limits=norm_params,
                                         zero_center=True,
                                         restore=True)
            Out2_tile = pred_E[0].data.numpy()  ##new!
                    
            #pdb.set_trace()
            #plt.show(plt.imshow(Out_tile,cmap="gray"))
            #print weights.shape, Out_tile.shape
            output_line[:,j:j+Out_tile.shape[1]] += weights * Out_tile
            output_weights[:,j:j+Out_tile.shape[1]] += weights
            
            for cl in range(0,n_cl):
            
                output2_line[:,j:j+Out_tile.shape[1],cl] += weights2[:,:,cl] * Out2_tile[cl]
                output2_weights[:,j:j+Out_tile.shape[1],cl] += weights2[:,:,cl]

            j += opt.fineSize - opt.overlap
        
        
        if i == (dataset.DSM.Rows - opt.fineSize):
            #pdb.set_trace()
            output_line /= output_weights
            for cl in range(0,n_cl):                
                output2_line[:,:,cl] /= output2_weights[:,:,cl] ##new!           
            
            Out.writeImageROI(full_line[(i-prev):opt.overlap,:],0,i)
            Out.writeImageROI(output_line[(prev+opt.overlap-i):,:],0,prev+opt.overlap)
            ##new!
            Out2.writeImageROI(full2_line[(i-prev):opt.overlap,:,:],0,i)
            Out2.writeImageROI(output2_line[(prev+opt.overlap-i):,:,:],0,prev+opt.overlap)
            #Out.writeImageROI(output_line,0,i))
            break
        else:

            output_line[:-opt.overlap,:] /= output_weights[:-opt.overlap,:]                             
            Out.writeImageROI(output_line[:-opt.overlap,:],0,i)
            ##new!
            for cl in range(0,n_cl):
                output2_line[:-opt.overlap,:,cl] /= output2_weights[:-opt.overlap,:,cl]         
            Out2.writeImageROI(output2_line[:-opt.overlap,:,:],0,i)
                        

        full_weights = output_weights.copy()
        full_line = output_line.copy()
        ##new!
        full2_weights = output2_weights.copy()
        full2_line = output2_line.copy()
        
        # shift for next line..
        output_line[:opt.overlap,:] = output_line[-opt.overlap:,:]
        output_line[opt.overlap:,:] = 0
        output_weights[:opt.overlap,:] = output_weights[-opt.overlap:,:]
        output_weights[opt.overlap:,:] = 0
        ##new!
        output2_line[:opt.overlap,:,:] = output2_line[-opt.overlap:,:,:]
        output2_line[opt.overlap:,:,:] = 0
        output2_weights[:opt.overlap,:,:] = output2_weights[-opt.overlap:,:,:]
        output2_weights[opt.overlap:,:,:] = 0
                
        prev = i
        
        i += opt.fineSize - opt.overlap
        
        
        
        if (i + opt.fineSize - opt.overlap) > (dataset.DSM.Rows - opt.overlap):   

            i = min(i, dataset.DSM.Rows - opt.fineSize)          
            
            #pdb.set_trace()
            output_line = np.zeros((opt.fineSize,Out.Columns), dtype=np.float32)
            output_line[:opt.overlap,:] = full_line[(i-prev):(i-prev)+opt.overlap:]
            output_line[opt.overlap:,:] = 0
            
            output_weights = np.zeros((opt.fineSize,Out.Columns), dtype=np.float32)
            output_weights[:opt.overlap,:] = full_weights[(i-prev):(i-prev)+opt.overlap:]
            output_weights[opt.overlap:,:]  = 0
            ##new!
            output2_line = np.zeros((opt.fineSize,Out2.Columns,n_cl), dtype=np.float32)
            output2_line[:opt.overlap,:,:] = full2_line[(i-prev):(i-prev)+opt.overlap:,:,:]
            output2_line[opt.overlap:,:,:] = 0
            
            output2_weights = np.zeros((opt.fineSize,Out2.Columns,n_cl), dtype=np.float32)
            output2_weights[:opt.overlap,:,:] = full2_weights[(i-prev):(i-prev)+opt.overlap:,:,:]
            output2_weights[opt.overlap:,:,:]  = 0

 
            #pdb.set_trace()                
        print "i", i

    #eg = xdibias.Image(os.path.join(opt.test_outdir,'result_edges'))
    #eg_data = eg.readImageData()
    #eg_result = eg_data.argmax(axis=2)
    #xdibias.imwrite(eg_result.astype('uint8'),os.path.join(opt.test_outdir,'result_edges_bin'),eg)
print("t_fwd:%.4f\n", t_fwd)    
