import os
import math
import numpy as np
import cv2
import warnings
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
    img = img.reshape((-1, step_h, step_w))

    oris =[]
    emag = []

    oris_vis_row=None
    oris_vis=None

    for ii, subimg in enumerate(img):
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


    oris = np.array(oris).reshape((height,width))
    emag = np.array(emag).reshape((height,width))

    return oris, emag

#==================================================================================#
def OriTuning(oris,emag,num_ori):

    ori_vals = oris.flatten()
    Oxy = np.tile(ori_vals,(len(ori_vals),1))

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
    Oij = np.pi + Oij                           #  每个点与其他所有点的方位关系（0--2*pi）
    Dij = np.sqrt(pow(dXij,2)+pow(dYij,2)) / np.sqrt(ww*ww + hh*hh)         # 每个点与其他所有点的距离关系（归一化）

    factor = len(ori_vals)-np.isnan(oris).sum()
    Sij = np.abs((np.pi/2-np.arctan(dYij/dXij))-Oxy)
    Sij[Sij>np.pi/2] = np.pi-Sij[Sij>np.pi/2]   # 每个点与其他所有点的夹角关系（对应点朝向与两点连线的夹角）

    tFAij = np.zeros((num_ori,Oij.shape[0],Oij.shape[1]))
    tFSij = np.zeros((num_ori,Sij.shape[0],Sij.shape[1]))
    dd = np.eye(Sij.shape[0])

    for ii in range(num_ori): 
        taij = np.zeros(Oij.shape)
        taij[np.where((Oij>=ii*2*np.pi/16) & (Oij<(ii+1)*2*np.pi/16))] = 1  # 筛选出特定方位（范围）的点

        taij = taij * Dij                                                   # 筛选出特定方位（范围）的点的距离

        taij[np.isnan(Oxy)]=np.inf
        taij[taij==0]=np.inf           # 除去0之外的最小值
        tmdis = np.min(taij,axis=1) 

        xidx = np.arange(0, ww*hh)
        yidx = np.argmin(taij, axis=1)
        yidx[np.isnan(tmdis)] = -1     # 去除没有碰到轮廓点的方向
        toij = np.zeros(Oij.shape)
        toij[xidx,yidx]=1              # 每个点在ii方位上，最近的轮廓点位置
        tFAij[ii,:,:] = toij

        tsij = np.zeros(Sij.shape)
        tsij[np.where((Sij>=ii*np.pi/32) & (Sij<=(ii+1)*np.pi/32))] = 1
        tsij = dd + tsij
        tFSij[ii,:,:] = tsij/factor     # 与当前点连线夹角在第ii范围内的连接矩阵

    FAij = tFAij/num_ori
    FSij = tFSij/num_ori

    return FAij, FSij                   # FAij--选择pooling外周轮廓点的连接矩阵； 
                                        #  FSij -- 与当前点连线夹角在第ii范围内的连接矩阵
  


#==================================================================================#
if __name__=='__main__':
    # import scipy.io as scio

    root='E:\ETHZ\LRSP\datasets'  
    imgs = list(sorted(os.listdir(os.path.join(root, "contourtest")))) #cmax  contourtest
    imgs_path = os.path.join(root, "contourtest", imgs[2])
    img = cv2.imread(imgs_path,1).astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.figure()
    plt.imshow(img)
    plt.pause(0.001) 
    plt.show()

    oris, emag = OriFitting(img, height=16, width=16)
    print(oris)
    plt.figure()
    plt.imshow(oris)
    plt.pause(0.001) 
    plt.show()
 
    aij = OriTuning(oris,emag, num_ori=16)
    plt.figure()
    plt.imshow(aij[2,:,:])
    plt.pause(0.001) 
    plt.show()
    print(aij.shape)

 
    


