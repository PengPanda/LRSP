import os
import numpy as np
# from PIL import Image
import math
import random
import cv2
from matplotlib import pyplot as plt
# from torch.autograd import Variable

import torch
import torchvision
# from torchvision import transforms

# from vputils import *
from orientation_column_linedrawing import OriFitting, OriTuning 

#==================================================================================#
# build dataset

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
NUM_ORI = 8  # 1  or  8
GRIDS = 16
IMSIZE = 256

class SALdataset(object):
    def __init__(self, root, train=True, test=False):
        self.root = root
        self.train = train
        self.test = test

        if self.train:
            self.imgs = list(sorted(os.listdir(os.path.join(self.root, "images/train"))))
            self.edge = list(sorted(os.listdir(os.path.join(self.root, "edges/train"))))      
            self.fixs = list(sorted(os.listdir(os.path.join(self.root, "maps/train"))))
        else:
            self.imgs = list(sorted(os.listdir(os.path.join(self.root, "images/val"))))
            self.edge = list(sorted(os.listdir(os.path.join(self.root, "edges/val"))))      
            self.fixs = list(sorted(os.listdir(os.path.join(self.root, "maps/val"))))

        if self.test:
            self.imgs = list(sorted(os.listdir(os.path.join(self.root, "images/test"))))
            self.edge = list(sorted(os.listdir(os.path.join(self.root, "edges/test"))))      

    def __getitem__(self, idx):
        # load images ad masks
        if self.train:
            imgs_path = os.path.join(self.root, "images/train", self.imgs[idx])
            edge_path = os.path.join(self.root, "edges/train", self.edge[idx])     
            fixs_path = os.path.join(self.root, "maps/train", self.fixs[idx])
        else:
            imgs_path = os.path.join(self.root, "images/val", self.imgs[idx])
            edge_path = os.path.join(self.root, "edges/val", self.edge[idx])     
            fixs_path = os.path.join(self.root, "maps/val", self.fixs[idx])

        if self.test:
            imgs_path = os.path.join(self.root, "images/test", self.imgs[idx])
            edge_path = os.path.join(self.root, "edges/test", self.edge[idx])     
        
        imname = self.imgs[idx]

        img = cv2.imread(imgs_path,1).astype('float32')
        # img = img-cv2.GaussianBlur(img, (51,51), 21) 
        img = img[:,:,::-1] 

        edge = cv2.imread(edge_path,1).astype('float32')
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        edge = edge * (1.0 / 255)

        fixs = cv2.imread(fixs_path,cv2.IMREAD_GRAYSCALE).astype('float32')
        fixs = fixs * (1.0 / np.max(fixs))

        img = cv2.resize(img,(IMSIZE,IMSIZE))
        edge = cv2.resize(edge,(IMSIZE,IMSIZE))
        fixs = cv2.resize(fixs,(IMSIZE,IMSIZE))    

        if self.train:
            img = np.clip(img, 0.0, 255.0)
            img = img * (1.0 / 255)
            img = (img-MEAN) / STD
            img, edge, fix_labels= self.augment_train(img, edge, fixs)

            rth = 0.2 #random.uniform(0.001, 0.3) 
            # edge[edge<=rth]=0
            # print(rth)

            oris, emag= OriFitting(edge, height=GRIDS, width=GRIDS, edge_th=rth)
            aij, sij = OriTuning(oris,emag,NUM_ORI*2)

            ori_labels = np.zeros(oris.shape)
            for ii in range(NUM_ORI):
                ori_labels[np.where((oris>=ii*np.pi/NUM_ORI) & (oris<=(ii+1)*np.pi/NUM_ORI))] = ii

        else:
            img = np.clip(img, 0.0, 255.0)
            img = img * (1.0 / 255)
            img = (img-MEAN) / STD
            fix_labels = fixs

            rth = 0.2
            oris, emag= OriFitting(edge, height=GRIDS, width=GRIDS, edge_th=rth)
            aij, sij = OriTuning(oris,emag,NUM_ORI*2)

            ori_labels = np.zeros(oris.shape)
            for ii in range(NUM_ORI):
                ori_labels[np.where((oris>=ii*np.pi/NUM_ORI) & (oris<=(ii+1)*np.pi/NUM_ORI))] = ii+1

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy())  
        img = img.type(torch.FloatTensor) 

        edge = torch.from_numpy(edge.copy()) 

        aij= torch.from_numpy(aij.copy())
        sij= torch.from_numpy(sij.copy())
        # aij = aij.unsqueeze(dim=0)  # 20210403 test


        # fdis = torch.from_numpy(fdis.copy())
        fix_labels = torch.from_numpy(fix_labels.copy())
        ori_labels= torch.from_numpy(ori_labels.copy())

        return img, edge, aij, sij, fix_labels, ori_labels, imname

   
    def augment_train(self, img, edge, fixs):

        flip_lr = random.randint(0, 1) 
        if flip_lr:
            img = img[:,::-1]
            edge = edge[:,::-1]
            fixs = fixs[:,::-1]

        flip_cov = random.randint(0, 1) 
        if flip_cov:
            rsx = int(random.randint(50, 100)/2)
            rsy = int(random.randint(50, 100)/2)
            rx, ry = random.randint(0, IMSIZE-rsx), random.randint(0, IMSIZE-rsy)
            # img[rx-rsx:rx+rsx-1, ry-rsy:ry+rsy-1] = 0
            edge[rx-rsx:rx+rsx-1, ry-rsy:ry+rsy-1] = 0

        # print(flip_cov)
        img = img.astype(np.float32)

        return img, edge, fixs
    
    def __len__(self):
        return len(self.imgs)

#===================================================================================#
# test the dataset

from matplotlib import pyplot as plt

if __name__=='__main__':

    # testdata = SALdataset(root='/scratch_net/moloch_second/ETHZ/torchcodes/structnet/data/KITTI/testing',train=True)
    testdata = SALdataset(root='E:/ETHZ/datasets/Salient/SALICON',train=True)
    dataload = torch.utils.data.DataLoader(testdata, batch_size=1,shuffle=False)  
    print(len(dataload))

    img, edge, aij, sij, fix_labels, ori_labels, imname = testdata[2]
    print(aij.shape)
    print(sij.shape)

    plt.figure()
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.pause(0.001) 
    plt.show()

    
    plt.figure()
    plt.imshow(edge)
    plt.pause(0.001) 
    plt.show()

    # plt.figure()
    # plt.imshow(fdis[0,:,:])
    # plt.pause(0.001) 
    # plt.show()

    plt.figure()
    plt.imshow(fix_labels)
    plt.pause(0.001) 
    plt.show()

    plt.figure()
    plt.imshow(ori_labels)
    plt.pause(0.001) 
    plt.show()
