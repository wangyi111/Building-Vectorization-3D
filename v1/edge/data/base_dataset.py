import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pdb

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
#    if opt.resize_or_crop == 'resize_and_crop':
#        osize = [opt.loadSize, opt.loadSize]
#        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
#        transform_list.append(transforms.RandomCrop(opt.fineSize))
#    elif opt.resize_or_crop == 'crop':
#        transform_list.append(transforms.RandomCrop(opt.fineSize))
#    elif opt.resize_or_crop == 'scale_width':
#        transform_list.append(transforms.Lambda(
#            lambda img: __scale_width(img, opt.fineSize)))
#    elif opt.resize_or_crop == 'scale_width_and_crop':
#        transform_list.append(transforms.Lambda(
#            lambda img: __scale_width(img, opt.loadSize)))
#        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.resize_or_crop == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=Image.BICUBIC)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())
        #transform_list.append(transforms.RandomResizedCrop(opt.fineSize, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2))

#    transform_list += [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5))]
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __scale_width(img, target_width):
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
