import numpy as np
from numpy.core.fromnumeric import shape
import torch
from matplotlib import pyplot as plt
import torchvision
import Dataset
import os
import cv2
import torch.nn.functional as F

from tqdm import tqdm
# import Dataset
from data_with_fix import Data
from edgenet_sharewgt import edgenet
from getNames import getNames
import time

torch.cuda.set_device(1)

class Visualize(object):
    def __init__(self,save_path) -> None:
        super(Visualize).__init__()
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def forward(self,features,name):

        # mean_fea = []
        for i in range(len(features)):
            fea = features[i]
            fea_img = torch.mean(torch.abs(fea),dim=1)
            fea_img = fea_img.squeeze().cpu().numpy() #cpu()
            fea_img = cv2.GaussianBlur(fea_img,(15,15),0)
            fea_img = self.normlize(fea_img)
            fea_img = 255*cv2.resize(fea_img, (256,256))
            # mean_fea.append(fea_img)
            fea_img = cv2.applyColorMap(fea_img.astype(np.uint8),cv2.COLORMAP_JET)
        
            img_name = self.save_path +name  + '_fea_' + str(i+1) + '.png'
            cv2.imwrite(img_name,fea_img)



        # mean_fea_img = np.mean(mean_fea,axis=0)
        # # mean_fea_img = cv2.GaussianBlur(mean_fea_img,(9,9),0)
        # mean_fea_img = cv2.applyColorMap(mean_fea_img.astype(np.uint8),cv2.COLORMAP_JET)
        # img_name = self.save_path + name +'_' + '_meanfea_' + '.png'
        # cv2.imwrite(img_name,mean_fea_img)

        # return fea_img
    
    def normlize(self,x):
        xmax = x.max()
        xmin = x.min()
        res = (x-xmin)/(xmax-xmin+0.000001)

        return res

def save_res(res_list,shape,name, save_path, key_word = 'prior',mod='Neg',abs=False,postpress = False):
    # res_list.unsqueeze(dim=0)
    for i,res in enumerate(res_list):
        if len(res.shape)<4:
            res = res.unsqueeze(dim=0)
        res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
        if abs:
            res = torch.mean(torch.abs(res),1)
        else:
            res = torch.mean(res,1)
        
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        if postpress:
            res = np.abs(np.mean(res)-res)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        if mod == 'Neg':
            res = 255 * (1-res)
        elif mod == 'Pos':
            res = 255 * (res)
        if i>=2:
            res = cv2.GaussianBlur(res,(15,15),0)
        else:
            res = cv2.GaussianBlur(res,(51,51),0)
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        res = cv2.applyColorMap(res.astype(np.uint8),cv2.COLORMAP_JET)

        # save_path  = os.path.join('./vis_feature/show_images/', dataset_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(save_path+'/'+name[0] +'_' + key_word +'_'+ str(i+1) + '_.png', res)
    



class Test(object):
    def __init__(self, Dataset, Network, dataset_name):
        ## dataset
        data_root   = './vis_feature/'
        self.path   = os.path.join(data_root, dataset_name)


        # self.cfg    = Dataset.Config(datapath=self.path, snapshot='./ckpts/edgenet/salient_rgb_model_50.pth', mode='test',batch_size=1)
        self.cfg    = Dataset.Config(datapath=self.path,dataset_name = dataset_name, snapshot='./ckpts/edgenet/salient_rgb_model_sharewgt_50.pth', mode='test',batch_size=1)
        # self.data   = Dataset.SALdataset(self.path, mode=self.cfg.mode)
        # getNames(path=data_root, folder=dataset_name,txt_name='test')
        self.data   = Data(self.cfg)
        self.dataset_name = dataset_name
        
        self.loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

        self.save_vis = Visualize(data_root)
    
    def save(self):
        with torch.no_grad():
            time_t = 0.0+1e-6

            for ii, datapack in enumerate(tqdm(self.loader), 0): # len(testdata)
                # img, edge, aij, fdis, ori_labels, imname, img_shape= datapack
                img, edge, aij, sij, shape, name= datapack  #SALICON
            
                edge = edge.unsqueeze(dim=0)
                # edge = edge.unsqueeze(dim=1)
                img = img.cuda().float()
                aij = aij.cuda().float()
                sij = sij.cuda().float()
                time_start = time.time()
                # res1,res2,res3,_,_,  _,_,_,_,_,  _,_,_,_= self.net(image)
                res1,lrsp1,lrsp2,before_lrsp = self.net(img,aij,sij)
                torch.cuda.synchronize()
                time_end = time.time()
                time_t = time_t + time_end - time_start
                # self.save_vis.forward(fea,name[0])   #save feature

                save_path = os.path.join('./vis_feature/show_images/','test_data')


                save_res(lrsp1,shape,name,save_path,key_word='lrsp1',mod='Pos',abs=True)
                save_res(lrsp2,shape,name,save_path,key_word='lrsp2',mod='Pos',abs=False,postpress = True)
                save_res(before_lrsp,shape,name,save_path,key_word='before_lrsp',mod='Pos',abs=True)
                # save_res(depth_fea,shape,name,save_path,key_word='depth_fea',mod='Pos')
                # save_res(rgb_fea,shape,name,save_path,key_word='rgb_fea',mod='Pos')

                # print('---done---')
                # for i, res in enumerate(fea):
                #     res = F.interpolate(res, shape, mode='bilinear', align_corners=True)
                #     res = res.sigmoid().data.cpu().numpy().squeeze()
                #     res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                #     res = 255 * (1-res)
                #     res = cv2.GaussianBlur(res,(15,15),0)
                #     res = cv2.applyColorMap(res.astype(np.uint8),cv2.COLORMAP_JET)
                #     save_path  = os.path.join('./vis_feature/show_images/', self.dataset_name)
                #     if not os.path.exists(save_path):
                #         os.makedirs(save_path)
                #     cv2.imwrite(save_path+'/'+name[0]+'_prior_'+ str(i)+'_.png', res)



            fps = len(self.loader) / time_t
            print('FPS is %f' %(fps))

# if __name__ == '__main__':
#     fea = []
#     for i in range(5):
#         fea.append(i*torch.randn(1,3,255,255))
#     save_root = './vis_feature/'

#     vis = Visualize(save_root)
#     vis.forward(fea)

if __name__=='__main__':
    # data_root = '/home/pp/WorkSpace/PythonSpace/pytorch/Datasets/'
    for data_path in ['test_data']: # 'ECSSD', 'PASCAL', 'HKUIS', 'DUTS-TE', 'DUTO'        
    # for data_path in ['SIP']:
        test = Test(Dataset, edgenet, data_path)
        test.save()
    
