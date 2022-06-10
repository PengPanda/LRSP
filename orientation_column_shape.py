import os
import math
import numpy as np
import cv2
import warnings
import copy
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skimage.measure import label

#==================================================================================#
warnings.simplefilter("ignore")

def OriFitting(img, height=32, width=32, edge_th=0.2):
    # img = cv2.resize(img,(256,256))
    # img[img<0.01]=0

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img * (1.0 / np.max(img))
    img_h, img_w= img.shape  # rows, cols
    step_h = int(img_h/height)
    step_w= int(img_w/width)
    img = cv2.resize(img,(step_w*width,step_h*height))

    img = np.lib.stride_tricks.as_strided(
    img,
    shape=(img_h // step_h, img_w // step_w, step_h, step_w),  # rows, cols
    strides=img.itemsize * np.array([step_h * img_w, step_w, img_w, 1])
    )

    # print(img.shape)
    img = img.reshape((-1, step_h, step_w))

    oris =[]
    emag = []

    oris_vis_row=None
    oris_vis=None

    for ii, subimg in enumerate(img):
        # plt.figure()
        # plt.imshow(subimg)
        # plt.pause(0.001) 
        # plt.show()

        # print(ii)

        emag.append(np.sum(subimg)/(step_h*step_w))

        # find the max connected region
        bw_img = subimg > edge_th
        labeled_img, num = label(bw_img, neighbors=8, background=0, return_num=True) 
        
        max_label = 0
        max_num = 0
        for i in range(1, num+1): 
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        lcc = (labeled_img == max_label)

        if num==0 or np.sum(lcc==1)<3:
            timg = np.zeros([step_h,step_w])
            oris.append(np.nan)
        else:
            idx = np.where(lcc==1)
            list_cnt = np.dstack((idx[1], idx[0])).squeeze()
            [vx,vy,x,y] = cv2.fitLine(list_cnt,cv2.DIST_L2,0,0.01,0.01)
            oris.append(np.pi/2-math.atan2(vy,vx))
            # print(math.atan(vy/vx))

    # #------------------------------------------------------------#
    #         timg = np.zeros([step_h,step_w])
    #         lefty = int((-x*vy/vx) + y)
    #         righty = int(((timg.shape[1]-x)*vy/vx)+y)
    #         timg = cv2.line(timg,(timg.shape[1]-1,righty),(0,lefty),255,2)
    #         # plt.figure()
    #         # plt.imshow(timg)
    #         # plt.pause(0.001) 
    #         # plt.show()

    #     if ii%width==0:
    #         oris_vis_row = timg
    #     else:
    #         oris_vis_row = np.hstack((oris_vis_row,timg))

    #     if ii>0 and ii%width==width-1:
    #         if ii==width-1:
    #             oris_vis = oris_vis_row
    #         else: 
    #             oris_vis = np.vstack((oris_vis,oris_vis_row))
    #         oris_vis_row = None
       
    # plt.figure()
    # plt.imshow(oris_vis)
    # plt.pause(0.001) 
    # plt.show()
    # #------------------------------------------------------------#

    oris = np.array(oris).reshape((height,width))
    emag = np.array(emag).reshape((height,width))

    return oris, emag

#==================================================================================#
def OriTuning(oris,emag,num_ori):

    ori_vals = oris.flatten()
    Oxy = np.tile(ori_vals,(len(ori_vals),1))
    # max_orivalue = np.max(ori_vals)

    emag_vals = emag.flatten()
    Exy = np.tile(emag_vals,(len(emag_vals),1))
    max_edgevalue = np.max(emag_vals)

    ww,hh = oris.shape[0], oris.shape[1]

    x = np.arange(0, ww)
    y = np.arange(0, hh)
    xx, yy = np.meshgrid(x, y)
    x_vals, y_vals = xx.flatten(), yy.flatten()
    Xdis = pdist(x_vals[:,np.newaxis], lambda u, v: v-u)   
    Ydis = pdist(y_vals[:,np.newaxis], lambda u, v: v-u)
    dXij = squareform(Xdis)
    dYij = squareform(Ydis)

    dXij = np.triu(dXij,1)-np.tril(dXij,-1)
    dYij = np.triu(dYij,1)-np.tril(dYij,-1)

    Oij = np.pi/2-np.arctan(dYij/dXij)
    Oij[dYij<0] = -Oij[dYij<0]
    Oij = np.pi + Oij               #  每个点与其他所有点的方位关系（0--2*pi）

    Dij = np.sqrt(pow(dXij,2)+pow(dYij,2)) / np.sqrt(ww*ww + hh*hh) # 每个点与其他所有点的距离关系（归一化）

    # plt.figure()
    # plt.imshow(Oij)
    # plt.pause(0.001) 
    # plt.show()

    # plt.figure()
    # plt.imshow(Dij)
    # plt.pause(0.001) 
    # plt.show()

    Fdis = np.zeros((num_ori,ww,hh))
    Foxy = np.zeros((num_ori,ww,hh))
    Fexy = np.zeros((num_ori,ww,hh))
    tFAij = np.zeros((num_ori,Oij.shape[0],Oij.shape[1]))
    tFCij = np.zeros((num_ori,Oij.shape[0],Oij.shape[1]))

    for ii in range(num_ori): 
        taij = np.zeros(Oij.shape)
        taij[np.where((Oij>=ii*2*np.pi/16) & (Oij<(ii+1)*2*np.pi/16))] = 1  # 筛选出特定方位（范围）的点
        taij = taij * Dij                                                   # 筛选出特定方位（范围）的点的距离

        taij0 = copy.deepcopy(taij)
        taij0[taij==0]=np.inf 

        taij[np.isnan(Oxy)]=np.inf
        taij[taij==0]=np.inf                    # 除去0之外的最小值
        tmdis = np.min(taij,axis=1) 

        # tmdis[np.isinf(tmdis)]=np.nan
        # for kk in range(ww*hh):
        #     idxx = taij0[kk,:]<tmdis[kk]
        #     tFCij[ii,kk,idxx] = 1          

        tmdis[np.isinf(tmdis)]=np.nan
        Fdis[ii,:,:] = tmdis.reshape((ww,hh))   # 每个点在ii方位上，最近的轮廓点距离，作为特征
        Fdis[np.isnan(Fdis)] = 0

        xidx = np.arange(0, ww*hh)
        yidx = np.argmin(taij, axis=1)
        yidx[np.isnan(tmdis)] = -1     # 去除没有碰到轮廓点的方向
        toij = np.zeros(Oij.shape)
        toij[xidx,yidx]=1              # 每个点在ii方位上，最近的轮廓点位置
        tFAij[ii,:,:] = toij

        
        toxy = Oxy[xidx,yidx]/np.pi
        texy = Exy[xidx,yidx]/max_edgevalue
        Foxy[ii,:,:] = toxy.reshape((ww,hh))    # 每个点在ii方位上，最近的轮廓点的朝向; 作为特征
        Fexy[ii,:,:] = texy.reshape((ww,hh))    # 每个点在ii方位上，最近的轮廓点的强度; 作为特征 
        Foxy[np.isnan(Foxy)] = 0
        Fexy[np.isnan(Fexy)] = 0

        # plt.figure()
        # plt.imshow(toxy.reshape((ww,hh)))
        # plt.pause(0.001) 
        # plt.show()

    # dd = np.eye(Oij.shape[0])
    # FAij = (dd + tFAij.sum(axis=0))/(num_ori+1)  # pooling surround contours 
    FAij = tFAij/num_ori
    Feats= np.concatenate((Fdis,Foxy,Fexy),axis=0)    

    return FAij,  Feats       # FAij--选择pooling外周轮廓点的连接矩阵；
  
    
#==================================================================================#
if __name__=='__main__':

    root='E:\ETHZ\Torch\LRSP\datasets'  
    imgs = list(sorted(os.listdir(os.path.join(root, "contourtest")))) #cmax  contourtest
    imgs_path = os.path.join(root, "contourtest", imgs[0])
    img = cv2.imread(imgs_path,1).astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.figure()
    plt.imshow(img)
    plt.pause(0.001) 
    plt.show()

    oris, emag= OriFitting(img, height=16, width=16)
    print(oris.shape)
    plt.figure()
    plt.imshow(oris)
    plt.pause(0.001) 
    plt.show()
 
    aij, Feats  = OriTuning(oris,emag, num_ori=16)
    plt.figure()
    plt.imshow(aij[2,:,:])
    plt.pause(0.001) 
    plt.show()
    print(aij.shape)

    plt.figure()
    plt.imshow(Feats[1,:,:])
    plt.pause(0.001) 
    plt.show()

    # plt.figure()
    # plt.imshow(Feats[17,:,:])
    # plt.pause(0.001) 
    # plt.show()

    # plt.figure()
    # plt.imshow(Feats[33,:,:])
    # plt.pause(0.001) 
    # plt.show()

    # print(Feats.shape)
    


