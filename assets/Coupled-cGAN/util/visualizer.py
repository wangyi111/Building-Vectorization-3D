import numpy as np
import os
import ntpath
import time
from . import util
from . import html

import pdb

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

"""  rescale image values  """
def intensity_rescaling(image):
    out = np.zeros_like(image)
    #pdb.set_trace()
    for i in range (0,image.shape[-1]):
        if image[:,:,i].min()-image[:,:,i].max() != 0: # new!!
            out[:,:,i] = (image[:,:,i]-image[:,:,i].min())*255/(image[:,:,i].max()-image[:,:,i].min())
    out[out<0] = 0
    out[out>255] = 255
    return out.astype(np.uint8)

       
class Visualizer():
    """  initialize dirs to save visualization results  """
    def __init__(self, opt):
        # self.opt = opt 
        self.display_id = opt.display_id # window id of the web display
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port) # create visdom object (run visdom on port first)
            self.display_single_pane_ncols = opt.display_single_pane_ncols # default: 0

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web') # web dir
            self.img_dir = os.path.join(self.web_dir, 'images') # images dir
            print('create web directory %s...' % self.web_dir)
            logger.info('Creating web directory at %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt') # loss_log dir
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    """  diplay images  """
    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        #pdb.set_trace()
        """ show images in the browser """
        if self.display_id > 0: 
        
            """ ??? """
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win = self.display_id + 2,
                              opts=dict(title=title + ' labels'))

            else:
                idx = 1
                #pdb.set_trace()
                for label, image_numpy in visuals.items():
                    #image_numpy = np.flipud(image_numpy)
                    # split two channel image into two separate images (DSM + Ortho)
                    if image_numpy.shape[2] == 2:
                        self.vis.image(image_numpy[:,:,0:1].transpose([2,0,1]), opts=dict(title=label + " Band 1"),
                                           win=self.display_id + idx)
                        idx += 1
                        self.vis.image(image_numpy[:,:,1:].transpose([2,0,1]), opts=dict(title=label + " Band 2"),
                                           win=self.display_id + idx)
                        idx += 1
                    else:
                        self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                           win=self.display_id + idx)
                        idx += 1
                        
        """ save images to disk and html file """
        if self.use_html: # save images to a html file
            # save images
            #pdb.set_trace()
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                if label == "real_O":
                    image_numpy = intensity_rescaling(image_numpy)
                if label == "real_E": # new!!
                    image_numpy = intensity_rescaling(image_numpy)
                if label == "real_I": # new!!
                    image_numpy = intensity_rescaling(image_numpy)                
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    
    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        #pdb.set_trace()
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        #self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['X'].append(epoch - 1 + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': 'loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id+7)
            
#    # errors: dictionary of error labels and values
#    def plot_current_weights(self, epoch, counter_ratio, opt, errors):
#        if not hasattr(self, 'plot_weights'):
#            self.plot_weights = {'X':[],'Y':[], 'legend':list(errors.keys())}
#        self.plot_weights['X'].append(epoch + counter_ratio)
#        self.plot_weights['Y'].append([errors[k] for k in self.plot_weights['legend']])
#        self.vis.line(
#            X=np.stack([np.array(self.plot_weights['X'])]*len(self.plot_weights['legend']),1),
#            Y=np.array(self.plot_weights['Y']),
#            opts={
#                'title': 'weights over time',
#                'legend': self.plot_weights['legend'],
#                'xlabel': 'epoch',
#                'ylabel': 'loss'},
#            win=self.display_id+6)


#    # errors: same format as |errors| of plotCurrentErrors
#    def print_current_errors(self, epoch, i, errors, t):
#        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
#        for k, v in errors.items():
#            message += '%s: %.3f ' % (k, v)
#
#        print(message)
#        with open(self.log_name, "a") as log_file:
#            log_file.write('%s\n' % message)

    # weights: dictionary of weight labels and values
    def plot_current_weights(self, epoch, counter_ratio, opt, weights):
        #pdb.set_trace()
        if not hasattr(self, 'plot_weights'):
            self.plot_weights = {'X':[],'Y':[], 'legend':list(weights.keys())}
        self.plot_weights['X'].append(epoch - 1 + counter_ratio)
        self.plot_weights['Y'].append([weights[k] for k in self.plot_weights['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_weights['X'])]*len(self.plot_weights['legend']),1),
            Y=np.array(self.plot_weights['Y']),
            opts={
                'title': 'weights over time',
                'legend': self.plot_weights['legend'],
                'xlabel': 'epoch',
                'ylabel': 'weights s = log(sigma^2)'},
            win=self.display_id+8)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        #pdb.set_trace()
        message = '(epoch: %d, iters: %d, time: %.3f) Losses: ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        logger.info(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_weights(self, epoch, i, errors, t):
        #pdb.set_trace()
        message = '(epoch: %d, iters: %d, time: %.3f) Task Weights: ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        logger.info(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            
    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
