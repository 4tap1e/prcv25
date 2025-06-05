from dataset.transform_3d import *
import numpy as np
import math, h5py
import random, time
import torch
from copy import deepcopy
from torch.utils.data import Dataset


class Flare3Dataset(Dataset):
    """ Flare2022 Dataset """
    def __init__(self, mode, args, size, nsample=None):
        self.dir = args.base_dir
        self.size = size
        self.mode = mode
        # self.n_class = args.nclass
        self.path = self.dir + '/{}.txt'.format(mode) 
        with open(self.path, 'r') as f:
            self.name_list = f.readlines()
        self.name_list = [item.replace('\n', '').split(",")[0] for item in self.name_list]
        
        if mode == 'train_u' and args.num is not None:
            self.name_list = self.name_list[:args.num]
            
        if mode == 'train_l' and nsample is not None and nsample > len(self.name_list):
            self.name_list *= math.ceil(nsample / len(self.name_list))
            self.name_list = self.name_list[:nsample]

    def __getitem__(self, idx):
        id = self.name_list[idx]
        start = time.time()
        if self.mode.split('_')[0] == 'train':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img = h5f['image'][:]
            if self.mode == 'train_l':
                mask = h5f['label'][:]
            elif self.mode == 'train_u':
                mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            
        if self.mode == 'val':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            img, mask = normalize_3d(img, mask)     
            return img, mask, idx 
        
        h5f.close()
        # img, mask = resize_3d(img, mask, (0.75, 2.0), self.n_class)
        ignore_value = 254 if self.mode == 'train_u' else 255
        # img, mask = flip_3d(img, mask)
        img, mask = crop_3d(img, mask, self.size, ignore_value)
        # img, mask = cropped4ct_3d(img, mask, self.size, ignore_value)
        
        if self.mode == 'train_l':
            return normalize_3d(img, mask)             
         
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        
        # if random.random() < 0.5:
        #     img_s1 = blur_3d(img_s1)
        if random.random() < 0.5:
            img_s1 = ColorJitter3d(img_s1)
        cutmix_box1 = obtain_cutmix_box_3d(img_s1)
        
        # if random.random() < 0.5:
        #     img_s2 = blur_3d(img_s2)
        if random.random() < 0.5:
            img_s2 = ColorJitter3d(img_s2)
        cutmix_box2 = obtain_cutmix_box_3d(img_s2)
        
        ignore_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
        img_s1, ignore_mask = normalize_3d(img_s1, ignore_mask)
        img_s2 = normalize_3d(img_s2)
        
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        return normalize_3d(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
    
    def __len__(self):
        return (len(self.name_list))


class Flare3Dataset_wtCJ(Dataset):
    """ Flare2022 Dataset """
    def __init__(self, mode, args, size, nsample=None):
        self.dir = args.base_dir
        self.size = size
        self.mode = mode
        # self.n_class = args.nclass
        self.path = self.dir + '/{}.txt'.format(mode) 
        with open(self.path, 'r') as f:
            self.name_list = f.readlines()
        self.name_list = [item.replace('\n', '').split(",")[0] for item in self.name_list]
        
        if mode == 'train_u' and args.num is not None:
            self.name_list = self.name_list[:args.num]
            
        if mode == 'train_l' and nsample is not None and nsample > len(self.name_list):
            self.name_list *= math.ceil(nsample / len(self.name_list))
            self.name_list = self.name_list[:nsample]

    def __getitem__(self, idx):
        id = self.name_list[idx]
        start = time.time()
        if self.mode.split('_')[0] == 'train':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img = h5f['image'][:]
            if self.mode == 'train_l':
                mask = h5f['label'][:]
            elif self.mode == 'train_u':
                mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            
        if self.mode == 'val':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            img, mask = normalize_3d(img, mask)     
            return img, mask, idx 
        
        h5f.close()
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop_3d(img, mask, self.size, ignore_value)
        if self.mode == 'train_l':
            return normalize_3d(img, mask)             
         
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        cutmix_box1 = obtain_cutmix_box_3d(img_s1)
        cutmix_box2 = obtain_cutmix_box_3d(img_s2)
        
        ignore_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
        img_s1, ignore_mask = normalize_3d(img_s1, ignore_mask)
        img_s2 = normalize_3d(img_s2)
        
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        return normalize_3d(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
    
    def __len__(self):
        return (len(self.name_list))


class Flare3Dataset_wtCJCM(Dataset):
    """ Flare2022 Dataset """
    def __init__(self, mode, args, size, nsample=None):
        self.dir = args.base_dir
        self.size = size
        self.mode = mode
        # self.n_class = args.nclass
        self.path = self.dir + '/{}.txt'.format(mode) 
        with open(self.path, 'r') as f:
            self.name_list = f.readlines()
        self.name_list = [item.replace('\n', '').split(",")[0] for item in self.name_list]
        
        if mode == 'train_u' and args.num is not None:
            self.name_list = self.name_list[:args.num]
            
        if mode == 'train_l' and nsample is not None and nsample > len(self.name_list):
            self.name_list *= math.ceil(nsample / len(self.name_list))
            self.name_list = self.name_list[:nsample]

    def __getitem__(self, idx):
        id = self.name_list[idx]
        start = time.time()
        if self.mode.split('_')[0] == 'train':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img = h5f['image'][:]
            if self.mode == 'train_l':
                mask = h5f['label'][:]
            elif self.mode == 'train_u':
                mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            
        if self.mode == 'val':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            img, mask = normalize_3d(img, mask)     
            return img, mask, idx 
        
        h5f.close()
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop_3d(img, mask, self.size, ignore_value)
        
        if self.mode == 'train_l':
            return normalize_3d(img, mask)             
         
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        cutmix_box1 = obtain_cutmix_box_3d(img_s1)
        cutmix_box2 = obtain_cutmix_box_3d(img_s2)
        
        ignore_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
        img_s1, ignore_mask = normalize_3d(img_s1, ignore_mask)
        img_s2 = normalize_3d(img_s2)
        
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        return normalize_3d(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
    
    def __len__(self):
        return (len(self.name_list))
    

class Flare3Dataset_wtCutmix(Dataset):
    """ Flare2022 Dataset """
    def __init__(self, mode, args, size, nsample=None):
        self.dir = args.base_dir
        self.size = size
        self.mode = mode
        # self.n_class = args.nclass
        self.path = self.dir + '/{}.txt'.format(mode) 
        with open(self.path, 'r') as f:
            self.name_list = f.readlines()
        self.name_list = [item.replace('\n', '').split(",")[0] for item in self.name_list]
        
        if mode == 'train_u' and args.num is not None:
            self.name_list = self.name_list[:args.num]
            
        if mode == 'train_l' and nsample is not None and nsample > len(self.name_list):
            self.name_list *= math.ceil(nsample / len(self.name_list))
            self.name_list = self.name_list[:nsample]

    def __getitem__(self, idx):
        id = self.name_list[idx]
        start = time.time()
        if self.mode.split('_')[0] == 'train':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img = h5f['image'][:]
            if self.mode == 'train_l':
                mask = h5f['label'][:]
            elif self.mode == 'train_u':
                mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            
        if self.mode == 'val':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            img, mask = normalize_3d(img, mask)     
            return img, mask, idx 
        
        h5f.close()
        # img, mask = resize_3d(img, mask, (0.75, 2.0), self.n_class)
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop_3d(img, mask, self.size, ignore_value)
        # img, mask = flip_3d(img, mask)
        
        if self.mode == 'train_l':
            return normalize_3d(img, mask)             
         
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        
        # if random.random() < 0.5:
        #     img_s1 = blur_3d(img_s1)
        if random.random() < 0.5:
            img_s1 = ColorJitter3d(img_s1)
        cutmix_box1 = obtain_cutmix_box_3d(img_s1)
        
        # if random.random() < 0.5:
        #     img_s2 = blur_3d(img_s2)
        if random.random() < 0.5:
            img_s2 = ColorJitter3d(img_s2)
        cutmix_box2 = obtain_cutmix_box_3d(img_s2)
        
        ignore_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
        img_s1, ignore_mask = normalize_3d(img_s1, ignore_mask)
        img_s2 = normalize_3d(img_s2)
        
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        return normalize_3d(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
    
    def __len__(self):
        return (len(self.name_list))
    
    

class AmosDataset(Dataset):
    """ Amos Dataset """
    def __init__(self, mode, args, size, nsample=None):
        self.dir = args.base_dir
        self.size = size
        self.mode = mode
        # self.n_class = args.nclass
        self.path = self.dir + '/{}.txt'.format(mode.split('_')[0]) 
        with open(self.path, 'r') as f:
            self.name_list = f.readlines()
        self.name_list = [item.replace('\n', '').split(",")[0] for item in self.name_list]
        
        if args.num:
            self.name_list_l = self.name_list[:args.num]
        # if mode == 'train_u' and args.num is not None:
            self.name_list_u = self.name_list[args.num:]
            
        if mode == 'train_l' and nsample is not None and nsample >= len(self.name_list_l):
            self.name_list_l *= math.ceil(nsample / len(self.name_list_l))
            self.name_list_l = self.name_list_l[:nsample]
            self.name_list = self.name_list_l
        elif mode == 'train_u':
            self.name_list = self.name_list_u
        elif mode != 'val':
            print("Error: mode should be val, train_l or train_u, not {}".format(mode))

    def __getitem__(self, idx):
        id = self.name_list[idx]
        start = time.time()
        if self.mode.split('_')[0] == 'train':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img = h5f['image'][:]
            if self.mode == 'train_l':
                mask = h5f['label'][:]
            elif self.mode == 'train_u':
                mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            
        if self.mode == 'val':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            img, mask = normalize_3d(img, mask)     
            return img, mask, idx 
        
        h5f.close()
        ignore_value = 254 if self.mode == 'train_u' else 255
        # img, mask = flip_3d(img, mask)
        img, mask = crop_3d(img, mask, self.size, ignore_value)
        
        if self.mode == 'train_l':
            return normalize_3d(img, mask)             
         
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        
        # if random.random() < 0.5:
        #     img_s1 = blur_3d(img_s1)
        # if random.random() < 0.5:
        #     img_s1 = ColorJitter3d(img_s1)
        cutmix_box1 = obtain_cutmix_box_3d(img_s1)
        
        # if random.random() < 0.5:
        #     img_s2 = blur_3d(img_s2)
        # if random.random() < 0.5:
        #     img_s2 = ColorJitter3d(img_s2)
        cutmix_box2 = obtain_cutmix_box_3d(img_s2)
        
        ignore_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
        img_s1, ignore_mask = normalize_3d(img_s1, ignore_mask)
        img_s2 = normalize_3d(img_s2)
        
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        return normalize_3d(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
    
    def __len__(self):
        return (len(self.name_list))


class AmosDataset_wtAug(Dataset):
    """ Amos Dataset """
    def __init__(self, mode, args, size, nsample=None):
        self.dir = args.base_dir
        self.size = size
        self.mode = mode
        # self.n_class = args.nclass
        self.path = self.dir + '/{}.txt'.format(mode.split('_')[0]) 
        with open(self.path, 'r') as f:
            self.name_list = f.readlines()
        self.name_list = [item.replace('\n', '').split(",")[0] for item in self.name_list]
        
        if args.num:
            self.name_list_l = self.name_list[:args.num]
        # if mode == 'train_u' and args.num is not None:
            self.name_list_u = self.name_list[args.num:]
            
        if mode == 'train_l' and nsample is not None and nsample >= len(self.name_list_l):
            self.name_list_l *= math.ceil(nsample / len(self.name_list_l))
            self.name_list_l = self.name_list_l[:nsample]
            self.name_list = self.name_list_l
        elif mode == 'train_u':
            self.name_list = self.name_list_u
        elif mode != 'val':
            print("Error: mode should be val, train_l or train_u, not {}".format(mode))

    def __getitem__(self, idx):
        id = self.name_list[idx]
        start = time.time()
        if self.mode.split('_')[0] == 'train':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img = h5f['image'][:]
            if self.mode == 'train_l':
                mask = h5f['label'][:]
            elif self.mode == 'train_u':
                mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            
        if self.mode == 'val':
            h5f = h5py.File(self.dir + "/{}/2022.h5".format(id), 'r')
            img, mask = h5f['image'][:], h5f['label'][:]
            img, mask = normalize_3d(img, mask)     
            return img, mask, idx 
        
        h5f.close()
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop_3d(img, mask, self.size, ignore_value)
        
        if self.mode == 'train_l':
            return normalize_3d(img, mask)             
         
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        cutmix_box1 = obtain_cutmix_box_3d(img_s1)
        cutmix_box2 = obtain_cutmix_box_3d(img_s2)
        
        ignore_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.uint8)
        img_s1, ignore_mask = normalize_3d(img_s1, ignore_mask)
        img_s2 = normalize_3d(img_s2)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        return normalize_3d(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2
    
    def __len__(self):
        return (len(self.name_list))
    









