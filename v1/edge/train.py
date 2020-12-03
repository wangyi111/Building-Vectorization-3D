import time
import pdb
import yaml 
import torch
import os
import random
import numpy as np

import logging

from options.train_options import TrainOptions
from data.xdibias_dataset import XdibiasDSMLoader
from models.models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from util import metrics
# import EarlyStopping
from util.pytorchtools import EarlyStopping

import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision
from sklearn.metrics import precision_score

"""  logger   """
logger = logging.getLogger(__name__)

logger.debug("Setting random seeds to zero!")
try:
    int(os.environ.get("PYTHONHASHSEED"))
except ValueError:
    raise Exception(
        "[ERROR] PYTHONHASHSEED must be set before starting python to" +
        " get reproducible results!"
    )

"""  fix random seeds to get reproduced experiments  """
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

"""  define and parse arguments  """
optparse = TrainOptions() ## from options.train_options import TrainOptions
optparse.parser.add_argument('--xd_loader', type=str, default='XdibiasDSMLoader',
                    help="xdibias data loader")
opt = optparse.parse() # <class argparse.Namespace>

"""  read dataset configuration  """
# read YAML/JSON config file
with open(os.path.join(opt.dataroot, "config.yaml")) as cf:
    config = yaml.load(cf) # <list>

loglevel = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL}
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=loglevel[opt.log_lvl.lower()])


"""  Load training dataset  """
#pdb.set_trace()
dataset_train = XdibiasDSMLoader(opt, config, config["data"]["roi_train"]) ### data.xdibias_dataset.py: Q3,Q4,Q5,Q6,Q7
   
train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size = opt.batchSize,
                                           shuffle = not opt.serial_batches,
                                           #num_workers = int(opt.nThreads)
                                           num_workers = 0
                                           )
n_samples_train = len(dataset_train)
logger.info('Got %d training images per epoch' % n_samples_train)

"""  Load validation dataset  """
#pdb.set_trace()
dataset_val = XdibiasDSMLoader(opt, config, config["data"]["roi_val"])    

val_loader = torch.utils.data.DataLoader(dataset_val,
                                           batch_size = opt.batchSize,
                                           shuffle = not opt.serial_batches,
                                           #num_workers = int(opt.nThreads)
                                           num_workers = 0
                                           )

n_samples_val = len(dataset_val)
logger.info('Got %d validation images' % n_samples_val)

"""  create model  """
#pdb.set_trace()                                            
model = create_model(opt) # models.models.py -> models.pix2pix_model.py: Q8,....

"""  create visualizes  """
visualizer = Visualizer(opt) # util.visualizer.py

"""  validation metrics  """
# set up validation metrics
val_metric = metrics.RMSE # metrics.py
best_metric = np.inf 
val_result_DSM = {}
val_result_DSM = np.inf
val_loss = {}
val_log = model.save_dir

"""  early stopping  """
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=config["patience"], verbose=True)
counter = 0

"""  tensorboard (new!)  """
log_dir = os.path.join(config["logging"]["log_dir"], 'unet_edge2')
logger.info("Writing tensorboard log to %s", log_dir)
tb_writer = SummaryWriter(log_dir)

"""  start training  """
logger.info("Starting training...")
total_steps = 0
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    logger.info("-------------------- Epoch " + str(epoch) + " --------------------")
    epoch_start_time = time.time()
    epoch_iter = 0


    '''
    ###################
    # train the model #
    ###################
    '''

    for data in tqdm(train_loader, desc="Current Epoch"): # iterate training data
        
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
        ## input a batch data ##
        #pdb.set_trace()        
        model.set_input(data)
         
        ## training ##
        #pdb.set_trace()
        model.optimize_parameters() # Optimize
        
        ## visualization ##
        # plot images        
        if total_steps % opt.display_freq == 0:
            #pdb.set_trace()
            visualizer.display_current_results(model.get_current_visuals(), epoch)
        # plot error/losses           
        if total_steps % opt.print_freq == 0:
            #pdb.set_trace()            
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/n_samples_train, opt, errors)
        # plot task weights
        '''
        if opt.loss_weights:
            if total_steps % opt.print_freq == 0:
                LossWeights = model.get_current_LossWeights()
                #pdb.set_trace()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_weights(epoch, epoch_iter, LossWeights, t)
                if opt.display_id > 0:
                    visualizer.plot_current_weights(epoch, float(epoch_iter)/n_samples_train, opt, LossWeights)
        '''                    
        ## tensorboard ##
        if total_steps % opt.display_freq == 0:
            #pdb.set_trace()
            #tb_writer.add_scalar(tag,scalar_value,global_step=None,walltime=None)
            #tb_writer.add_scalars(tag,scalar_dict,global_step=None,walltime=None)
            #tb_writer.add_image(tag,img_tensor,global_step=None,walltime=None,dataformats='CHW')
            errors = model.get_current_errors()
            tb_writer.add_scalars('losses',errors,global_step=total_steps,walltime=None)            
            LossWeights = model.get_current_LossWeights()
            tb_writer.add_scalars('loss_weights',LossWeights,global_step=total_steps,walltime=None)
            
            # add images seperately
            images = model.get_current_visuals()
            #tb_writer.add_image('stereo_dsm',images['real_A'],global_step=total_steps,walltime=None,dataformats='HWC')
            tb_writer.add_image('orthophoto',images['real_O'],global_step=total_steps,walltime=None,dataformats='HWC')
            #tb_writer.add_image('refined_dsm',images['fake_B'],global_step=total_steps,walltime=None,dataformats='HWC')
            tb_writer.add_image('GroundTruth_dsm',images['real_B'],global_step=total_steps,walltime=None,dataformats='HWC')
            tb_writer.add_image('predicted_corepoints',images['pred_E'],global_step=total_steps,walltime=None,dataformats='HWC')
            tb_writer.add_image('GroundTruth_corepoints',images['real_E'],global_step=total_steps,walltime=None,dataformats='HWC')
            #tb_writer.add_image('predicted_instances',images['fake_I'],global_step=total_steps,walltime=None,dataformats='HWC')
            #tb_writer.add_image('GroundTruth_instances',images['real_I'],global_step=total_steps,walltime=None,dataformats='HWC')
            
            # add imags in a table
            image_list = []
            for label,img in images.items():
                if img.shape[2]==1:
                    img = np.concatenate((img,)*3,axis=-1) # [H,W,1] to [H,W,3]
                img = np.transpose(img,(2,0,1)) # [C,H,W]
                image_list.append(torch.FloatTensor(img))
            #image_grid = torchvision.utils.make_grid(tensor,nrow=8,padding=2,normalize=False,range=None,scale_each=False,pad_value=0)
            image_grid = torchvision.utils.make_grid(image_list)
            tb_writer.add_image('images',image_grid,global_step=total_steps,walltime=None,dataformats='CHW')
            
        
############################# stop here ############################################### 
                
        '''
        ###########################    
        # mid-training validation #
        ###########################
        '''

        if total_steps % opt.val_freq == 0:
            
            logger.info("Pausing training for validation")

            # set model to evaluation mode (disable dropout, bn, and gradients)
            model.eval()
            
            with torch.no_grad():

                # reset metrics
                val_results_DSM = 0
                val_results_E = 0
                edge_cre = 0
                # run a forward pass
                for data in tqdm(val_loader, desc="Validation   "):
                    
                    sample = []

                    input_dsm = data["A"].cuda()
                    sample.append(input_dsm)
                    

                    if "Ortho" in data.keys():
                        input_img = data["Ortho"].cuda()
                        sample.append(input_img)
                        
                    
                    #pred_DSM,pred_E = model.netG.forward(*sample)
                    pred_E = model.netG.forward(*sample)
                    #pdb.set_trace()
                    #pred_E = torch.nn.Softmax(dim=1)(pred_E)

                    #dsm_RMSE += metrics.RMSE(pred_DSM, data["B"].cuda())
	            #dsm_MAE += metrics.MAE(pred_DSM, data["B"].cuda())

	            target_E = torch.argmax(data["Edges"],dim=1)
                    pred_label_E = torch.argmax(pred_E,dim=1).cpu()
	            edge_cre += torch.nn.CrossEntropyLoss()(pred_E,target_E.cuda().long()).item() 	      	            
	            #edge_precision += precision_score(target_E.squeeze(1).flatten(),pred_label_E.squeeze(1).flatten(),average='macro')
	    
	        # average results taking into account the batch size
	        #dsm_RMSE = dsm_RMSE / (len(dataset_val)/opt.batchSize)
	        #dsm_MAE = dsm_MAE / (len(dataset_val)/opt.batchSize)
	        edge_cre = edge_cre / (len(dataset_val)/opt.batchSize)
	        #edge_precision = edge_precision / (len(dataset_val)/opt.batchSize)
	        
                #val_result_DSM = dsm_RMSE
                # tensorboard
                #val_loss['dsm_RMSE'] = dsm_RMSE
                #val_loss['dsm_MAE'] = dsm_MAE
                val_loss['edge_cre'] = edge_cre
                #val_loss['edge_precision'] = edge_precision
                tb_writer.add_scalars('val_losses',val_loss,global_step=total_steps,walltime=None)

                # output results
#	        message = '[epoch: %d] test_errors: DSM_RMSE %.4f, DSM_MAE %.4f; edge_CRE %.4f, edge_precision %.4f' % (epoch,dsm_RMSE,dsm_MAE,edge_cre,edge_precision)
#	        with open(val_log, "a") as log_file:
#	            log_file.write('%s\n' % message)  
#                logger.info("---------- Validation Results ----------")
#                logger.info(message)


#            # early_stopping needs the validation loss to check if it has decresed, 
#            # and if it has, it will make a checkpoint of the current model
#            early_stopping(val_result, model)
#
#            if early_stopping.early_stop:
#                logger.info("---------- Early stopping ----------")
#                early_stopping.save_checkpoint(val_result, model, total_steps, epoch)
#                break

            model.train() # back to training

        best = val_result_DSM < best_metric           
        #pdb.set_trace()    
        if best:

            if opt.loss_weights:
                state = {
                    "iteration": total_steps,
                    "epoch": epoch,
                    "modelG_states": model.netG.state_dict(),
                    "loss_weights":  model.task_weights,
                    "optimizer_G": model.optimizer_G.state_dict()                    
                }
            else:
                state = {
                    "iteration": total_steps,
                    "epoch": epoch,
                    "modelG_states": model.netG.state_dict(),
                    "optimizer_G": model.optimizer_G.state_dict()
                }

            save_path = model.save_dir
            torch.save(state, save_path + "/best.pkl")
            logger.info("Updated checkpoint with best performance: (epoch %d, total_steps %d)" % (epoch, total_steps))

            
        best_metric = min(best_metric, val_result_DSM)

    if total_steps % opt.val_freq == 0:
        errors = model.get_current_errors()
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)

    if epoch % opt.save_epoch_freq == 0:
        logger.info('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    if epoch > opt.niter:
        model.update_learning_rate()
tb_writer.flush()
tb_writer.close()
