import time
import pdb
import yaml 
import torch
import os
import random
import numpy as np
import metrics
import logging

from options.train_options import TrainOptions
from data.xdibias_dataset import XdibiasDSMLoader
from models.models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm

# import EarlyStopping
from pytorchtools import EarlyStopping

import argparse


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
torch.manual_seed(SEED) ### Q1: set twice?
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)
np.random.seed(SEED) ### Q2: set twice?

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
pdb.set_trace()                                            
model = create_model(opt) # models.models.py  models.pix2pix_model.py: Q8,....

############################# stop here ###############################################
"""  create visualizes  """
visualizer = Visualizer(opt) # util.visualizer.py
total_steps = 0

"""  validation metrics  """
# set up validation metrics
val_metric = metrics.RMSE # metrics.py
best_metric = np.inf 
val_result = {}
val_result = np.inf

"""  early stopping  """
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=config["patience"], verbose=True)
counter = 0

#log_dir = os.path.join(cfg["logging"]["log_dir"], opt.name)
#logger.info("Writing tensorboard log to %s", log_dir)
#os.makedirs(log_dir, exist_ok=True)
#tb_writer = SummaryWriter(log_dir)

"""  start training  """
logger.info("Starting training...")

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    logger.info("-------------------- Epoch " + str(epoch) + " --------------------")
    epoch_start_time = time.time()
    epoch_iter = 0


    '''
    ###################
    # train the model #
    ###################
    '''

    for data in tqdm(train_loader, desc="Current Epoch"):

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
        ## input a batch data ##        
        model.set_input(data)
         
        ## training ##
        model.optimize_parameters() # Optimize
        
        ## visualization ##
        if total_steps % opt.display_freq == 0:
        #if total_steps % (len(dataset_val)/opt.batchSize) == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/n_samples_train, opt, errors)

        # plot task weights to visdom
        if opt.loss_weights:
            if total_steps % opt.print_freq == 0:
                LossWeights = model.get_current_LossWeights()
                #pdb.set_trace()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_weights(epoch, epoch_iter, LossWeights, t)
                if opt.display_id > 0:
                    visualizer.plot_current_weights(epoch, float(epoch_iter)/n_samples_train, opt, LossWeights)
                
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
                val_results = 0
                
                # run a forward pass
                for data in tqdm(val_loader, desc="Validation   "):
                    
                    sample = []

                    input_dsm = data["A"].cuda()
                    sample.append(input_dsm)
                    

                    if "Ortho" in data.keys():
                        input_img = data["Ortho"].cuda()
                        sample.append(input_img)
                        
                    
                    prediction = model.netG.forward(*sample)
                    #print (val_metric(prediction, data["B"].cuda()))

                    val_results += val_metric(prediction, data["B"].cuda())
                    
                    
                # average results taking into account the batch size
                #pdb.set_trace()
                val_result = val_results / (len(dataset_val)/opt.batchSize)

                # output results
                logger.info("---------- Validation Results ----------")
                logger.info("RMSE: %f", val_result)


#            # early_stopping needs the validation loss to check if it has decresed, 
#            # and if it has, it will make a checkpoint of the current model
#            early_stopping(val_result, model)
#
#            if early_stopping.early_stop:
#                logger.info("---------- Early stopping ----------")
#                early_stopping.save_checkpoint(val_result, model, total_steps, epoch)
#                break

            model.train()

        best = val_result < best_metric           
            
#        if total_steps % opt.save_latest_freq == 0 or best:
#
#            logger.info('saving the latest model (epoch %d, total_steps %d)' %
#                  (epoch, total_steps))
#                  
#            #model.save('latest')
#            if opt.loss_weights:
#                state = {
#                    "iteration": total_steps,
#                    "epoch": epoch,
#                    "modelG_states": model.concat_netG.state_dict(),
#                    "modelD_states": model.netD.state_dict(),
#                    "loss_weights":  model.task_weights,
#                    "optimizer_G": model.optimizer_G.state_dict(),
#                    "optimizer_D": model.optimizer_D.state_dict()
#                    
#                }
#            else:
#                state = {
#                    "iteration": total_steps,
#                    "epoch": epoch,
#                    "modelG_states": model.concat_netG.state_dict(),
#                    "modelD_states": model.netD.state_dict(),
#                    "optimizer_G": model.optimizer_G.state_dict(),
#                    "optimizer_D": model.optimizer_D.state_dict()
#                }
#
#            save_path = model.save_dir
#
#            if best:
#                
#                torch.save(state, save_path + "/best.pkl")
#                logger.info("Updated checkpoint with best performance")
#            
#
#            torch.save(state,
#                       save_path + "/iter_" + str(total_steps).zfill(8) + ".pkl")
#            logger.info("Saved checkpoint at iteration %i", total_steps)

            
        best_metric = min(best_metric, val_result)

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
