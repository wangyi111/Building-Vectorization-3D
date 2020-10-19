import pdb
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        #from .pix2pix_model_LossTrainedParam import Pix2PixModel
        pdb.set_trace()
        model = Pix2PixModel() # pix2pix_model.py
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
        
    pdb.set_trace()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
