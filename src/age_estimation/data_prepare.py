"""
This script prepares a dataset for facial age estimation.
@Author: Shichao (Nicholas) Li
Contact: nicholas.li@connect.ust.hk
"""
import torch
import torch.utils.data
import torchvision.transforms.functional as transform_f
import imageio as io
import numpy as np
import logging
# use PIL
import PIL


class FacialAgeDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, opt, split):
        self.dict = dictionary
        self.name = opt.dataset_name
        self.split = split
        self.image_size = opt.image_size
        self.crop_size = opt.crop_size
        self.crop_limit = self.image_size - self.crop_size
        assert self.name in ['FGNET', 'CACD', 'Morph'], 'Dataset not supported yet.'        
        self.cache = opt.cache
        self.image_channel = 1 if opt.gray_scale else 3
        self.transform = opt.transform
        self.img_path_list = []
        self.label = []
        self.scale_factor = 100
        if self.name == 'FGNET':
            self.mean = [0.425,  0.342,  0.314]
            self.std = [0.218,  0.191,  0.182]
            for key in dictionary:
                if len(key) == 3:
                    self.img_path_list += dictionary[key]['path']
                    self.label += dictionary[key]['age_list']
        elif self.name == 'CACD':
            self.mean = [0.432, 0.359, 0.320]
            self.std = [0.30,  0.264,  0.252]            
            for key in dictionary:
                self.img_path_list += dictionary[key]['path']
                self.label += dictionary[key]['age_list']
        elif self.name == 'Morph':
            self.mean = [0.564, 0.521, 0.508]
            self.std = [0.281,  0.255,  0.246]    
            self.img_path_list = dictionary['path']
            self.label = dictionary['age_list']            
        logging.info('{:s} {:s} set contains {:d} images'.format(self.name, 
                     self.split, len(self.img_path_list)))                  
        self.label = torch.FloatTensor(self.label)
        self.label /= self.scale_factor
        if opt.visualize:
            self.visualize()
            
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        if self.cache:
            raise NotImplementedError
        else:
            # read image from the disk
            image_path = self.img_path_list[idx]
            #image = io.imread(image_path)
            image = PIL.Image.open(image_path)
            # transformation for data augmentation
            if self.transform:
                # Use PIL and image augmentation provided by Pytorch
                if np.random.rand() > 0.5 and self.split == 'train':
                    image = transform_f.hflip(image)
                # only crop if input image size is large enough
                if self.crop_limit > 1:
                    # random cropping
                    if self.split == 'train':
                        x_start = int(self.crop_limit*np.random.rand())
                        y_start = int(self.crop_limit*np.random.rand())
                    else:
                        # only apply central-crop for evaluation set
                        x_start = 15
                        y_start = 15
                    image = transform_f.crop(image, y_start, x_start, 
                                             self.crop_size,
                                             self.crop_size)
            image = transform_f.to_tensor(image)
            image = transform_f.normalize(image, mean=self.mean, 
                                          std=self.std)                
        sample = {'image': image, 
                  'age': self.label[idx], 
                  'index': idx}    
        return sample
    
    def convert(self, img):
        # convert grayscale to RGB image if needed
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)    
        return img
    
    def get_label(self, idx):
        # this function only returns the label (avoid image reading)
        return self.label[idx]
    
    def get_image(self, idx):
        # only returns the raw image
        image_path = self.img_path_list[idx]
        image = io.imread(image_path)
        return image

def prepare_db(opt):
    # Prepare a list of datasets for training and evaluation
    train_list = []
    eval_list = []
    if opt.dataset_name == "FGNET":
        raise NotImplementedError
    elif opt.dataset_name == "CACD":
        eval_dic = np.load('../data/CACD_split/test_cacd_processed.npy').item()
        if opt.cacd_train:
            train_dic = np.load('../data/CACD_split/train_cacd_processed.npy').item()
            logging.info('Preparing CACD dataset (training with the training set).')
        else:
            train_dic = np.load('../data/CACD_split/valid_cacd_processed.npy').item()
            logging.info('Preparing CACD dataset (training with the validation set).')
        train_list.append(FacialAgeDataset(train_dic, opt, 'train'))
        eval_list.append(FacialAgeDataset(eval_dic, opt, 'eval'))
        return {'train':train_list, 'eval':eval_list}
    elif opt.dataset_name == "Morph":
        raise ValueError
    else:
        raise NotImplementedError