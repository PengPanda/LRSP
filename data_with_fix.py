import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from orientation_column_linedrawing import OriFitting, OriTuning
cv2.setNumThreads(0)
########################## Hyperparameters ##############################
NUM_ORI = 8  # 1  or  8
GRIDS = 16
IMSIZE = 256

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, edge=None, mask=None):
        image = (image - self.mean)/self.std
        edge /=255.0
        if mask is None:
            return image,edge
        else:
            mask /= 255.0
            return image, mask, edge

class RandomCrop(object):
    def __call__(self, image, mask, edge):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], edge[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask, edge):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1], edge[:, ::-1]
        else:
            return image, mask, edge

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, edge=None, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge  = cv2.resize( edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image,edge
        else:
            mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            # edge  = cv2.resize( edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
            return image, mask, edge

class ToTensor(object):
    def __call__(self, image, edge = None, aij=None, sij=None, mask=None):
        image = torch.from_numpy(image.copy()).float()
        image = image.permute(2, 0, 1)

        edge = torch.from_numpy(edge.copy()).float()
        # edge = edge.unsqueeze(0)
        aij= torch.from_numpy(aij).float()
        sij = torch.from_numpy(sij).float()

        if mask is None:
            return image,edge,aij,sij
        else:
            mask  = torch.from_numpy(mask.copy()).float()
            # edge  = torch.from_numpy(edge)
            return image, edge,aij,sij, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(256, 256)
        self.totensor   = ToTensor()
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())
        # self.samples = self.samples[:32] #testfortrain

    def __getitem__(self, idx):
        name  = self.samples[idx]
        ##---------------------train-----------
        if self.cfg.mode == 'train':
            image = cv2.imread(self.cfg.datapath+'/images/'+ self.cfg.mode+'/'+name+'.jpg')[:,:,::-1].astype(np.float32)
            edge = cv2.imread(self.cfg.datapath+'/edges/'+ self.cfg.mode +'/'+ name+'.jpg', 0).astype(np.float32)
        else:
            if self.cfg.dataset_name ==  'MIT1003':
                ##-----MIT1003-------------------------
                image = cv2.imread(self.cfg.datapath+'/imgs/' +name+'.jpeg')[:,:,::-1].astype(np.float32) #MIT1003
                edge = cv2.imread(self.cfg.datapath+'/edges/'+ name+'.jpeg', 0).astype(np.float32)
                ##------------------------------------
            elif self.cfg.dataset_name == 'MIT300':
                image = cv2.imread(self.cfg.datapath+'/imgs/' +name+'.jpg')[:,:,::-1].astype(np.float32)
                edge = cv2.imread(self.cfg.datapath+'/edges/'+ name+'.jpg', 0).astype(np.float32)

            elif self.cfg.dataset_name == 'Toronto':
                image = cv2.imread(self.cfg.datapath+'/imgs/' +name+'.jpg')[:,:,::-1].astype(np.float32)
                edge = cv2.imread(self.cfg.datapath+'/edges/'+ name+'.jpg', 0).astype(np.float32)
                
            elif self.cfg.dataset_name == 'DaytimeDataset':
                image = cv2.imread(self.cfg.datapath+'/imgs/' +name+'.png')[:,:,::-1].astype(np.float32)
                edge = cv2.imread(self.cfg.datapath+'/edges/'+ name+'.png', 0).astype(np.float32)

            elif self.cfg.dataset_name == 'FLIR':  ## for visualization
                image = cv2.imread(self.cfg.datapath+'/imgs/' +name+'.png')[:,:,::-1].astype(np.float32)
                edge = cv2.imread(self.cfg.datapath+'/edges/'+ name+'.png', 0).astype(np.float32)

            elif self.cfg.dataset_name == 'test_data':  ## for visualization
                image = cv2.imread(self.cfg.datapath+'/imgs/' +name+'.jpeg')[:,:,::-1].astype(np.float32)
                edge = cv2.imread(self.cfg.datapath+'/edges/'+ name+'.jpeg', 0).astype(np.float32)
            else:
                print('====  Wrong dataset, please check!  ======')



        shape = image.shape[:2]
        if self.cfg.mode=='train':
            mask  = cv2.imread(self.cfg.datapath + 'maps/' + 'train/' + name+'.png', 0).astype(np.float32)
            image,mask,edge = self.normalize(image,mask,edge)
            image,mask,edge = self.randomcrop(image,mask,edge)
            image,mask,edge= self.randomflip(image,mask,edge)
            image,mask,edge= self.resize(image,mask,edge)

            ##-------------------lrsp-------------
            rth = 0.2
            oris, emag= OriFitting(edge, height=GRIDS, width=GRIDS, edge_th=rth)
            aij,sij = OriTuning(oris,emag,NUM_ORI*2)

            ##==============================
            image,edge,aij,sij,mask= self.totensor(image,edge,aij,sij,mask)
            return image, mask, edge, aij, sij
        else:
            image,edge = self.normalize(image,edge)
            image,edge= self.resize(image,edge)   
            ##----------------lrsp---------------------
            rth = 0.2
            oris, emag= OriFitting(edge, height=GRIDS, width=GRIDS, edge_th=rth)
            aij,sij = OriTuning(oris,emag,NUM_ORI*2)

            ##========================================
            image,edge,aij,sij = self.totensor(image,edge,aij,sij)
            
            return image, edge ,aij,sij, shape, name


    def __len__(self):
        return len(self.samples)