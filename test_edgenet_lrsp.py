# # test the dataset
import numpy as np
from numpy.core.fromnumeric import shape
import torch
from matplotlib import pyplot as plt
from torch.utils.data import dataset
import torchvision
import Dataset
import os
import cv2
import torch.nn.functional as F

from tqdm import tqdm
from data_with_fix import Data
from edgenet_sharewgt import edgenet
from getNames import getNames

torch.cuda.set_device(1)


def Test(Dataset, Network, dataset_name):
    # dataset_root_path = '/home/pp/Datasets/FixationDataset/'
    dataset_root_path = '/home/pp/Datasets/FixationDataset/' #SALICON

    cfg_test = Dataset.Config(datapath=os.path.join(dataset_root_path,dataset_name),
    dataset_name=dataset_name,        
                               snapshot='./ckpts/edgenet/salient_rgb_model_sharewgt_50.pth',
                               mode='test', #mode='test', #SALICON
                               batch_size=1
                               )
    getNames(path=cfg_test.datapath, folder='imgs/',txt_name='test')
    dataset_test = Data(cfg_test)

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=16)  # num_workers=4
    print('len_dataset_test:', len(dataset_test))

    model = Network(cfg_test).cuda()
    model.eval()

    for ii, datapack in enumerate(tqdm(dataloader_test)): # len(testdata)
        img, edge ,aij,sij, img_shape, imname = datapack
    
        edge = edge.unsqueeze(dim=0)
        edge = edge.cuda().float()
        aij = aij.cuda().float()
        sij = sij.cuda().float()

        img = img.cuda().float()

        out = model(img,aij,sij) # img

        res = F.interpolate(out, (img_shape[0].item(),img_shape[1].item()), mode='bilinear', align_corners=True)
        res = sigmoid(res.sigmoid().data.cpu().numpy().squeeze())
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = 255 * res
       
        save_path  = './SaliencyMaps/EdgeNet_sharewgt/'+ cfg_test.dataset_name

        if not os.path.exists(save_path):
                    os.makedirs(save_path)
        cv2.imwrite(save_path+'/'+imname[0]+'.png', res)

def sigmoid(x):
# TODO: Implement sigmoid function
    return 1.0/(1 + np.exp(-x+0.75))

if __name__ == '__main__':
    dataset_names = ['FLIR']  # set your paths

    for i, name in enumerate(dataset_names):
        Test(Dataset, edgenet, name)

    print("That's it! Done...")


