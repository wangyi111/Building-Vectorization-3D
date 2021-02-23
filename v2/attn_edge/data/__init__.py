import torch.utils.data

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':            
        if opt.dataset_loader == "DSM2LOD":
            from data.xdibias_dataset import XdibiasDSMLoader
            dataset = XdibiasDSMLoader(opt)
        else:
            from data.aligned_dataset import AlignedDataset
            dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    #dataset.initialize(opt)
    #pdb.set_trace()
    return dataset

def create_dataset(opt):
    """Create a dataset given the option.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatasetDataLoader():
    
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
            
    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)
            
    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data

