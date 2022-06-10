import os
import torch
import datetime
import torch.optim as optim
import torchvision
from tqdm import tqdm

import Dataset
from data_with_fix import Data
from edgenet_sharewgt import edgenet
from utils import clip_gradient, kl_loss
from apex import amp

torch.cuda.set_device(1)
#==================================================================================#


def Train(Dataset, Network):

    cfg_train = Dataset.Config(datapath='/home/pp/Datasets/SALICON/',
                               savepath='./ckpts/edgenet/',
                               mode='train',
                               total_epoch=50,
                               batch_size=12, 
                               lr=0.001, #0.05 
                               momen=0.9, 
                               decay=5e-4
                               )
    if not os.path.exists(cfg_train.savepath):
        os.makedirs(cfg_train.savepath)
    ck_imgpath = './ckimgs/'

    # -----train----------
    dataset_train = Data(cfg_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=12, shuffle=True, num_workers=16)  # num_workers=4
    print('len_dataset_train:', len(dataset_train))

    

    #==================================================================================#
    # 2. model ...
    # #load a model pre-trained on ImageNet
    model = Network(cfg_train)
    model = model.cuda()


    # ========= Adam ==========
    # train_lr = 0.001
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr= cfg_train.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    #==========================
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    #==================================================================================#
    # 4. training...
    step_now = 0
    for epoch in range(cfg_train.total_epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg_train.total_epoch+1)*2-1))*cfg_train.lr
        model.train()
        running_loss = 0.0
        
        for i_batch, data in enumerate(tqdm(dataloader_train)):
            img, fix_labels, edge, aij, sij = data

            img = img.cuda().float()
            edge = edge.cuda().float()
            aij = aij.cuda().float()
            sij = sij.cuda().float()
            fix_labels = fix_labels.cuda().float()

            edge = edge.unsqueeze(dim=1)
            fix_labels = fix_labels.unsqueeze(dim=1)
            out = model(img, aij, sij) # RGB: img

            loss = kl_loss(out, fix_labels)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()

            optimizer.step()
            running_loss += loss.item()
            step_now += 1

            if i_batch%40 == 0:

                torchvision.utils.save_image(img, ck_imgpath + 'img'+str(epoch)+'_image.png', nrow=6, padding=2, normalize=False, range=None, scale_each=False)
                torchvision.utils.save_image(out, ck_imgpath + 'img'+str(epoch)+'_res.png', nrow=6, padding=2, normalize=False, range=None, scale_each=False)
                torchvision.utils.save_image(fix_labels, ck_imgpath + 'img'+str(epoch)+'_map.png', nrow=6, padding=2, normalize=False, range=None, scale_each=False)
        
        print('Epoch %d || Train Loss: %.4f' %
              (epoch+1, running_loss/len(dataloader_train)))

        with open("log.txt", "a") as f:
            f.write('%s | step:%d/%d |  epoch_loss = %.6f' %
                    (datetime.datetime.now(), epoch+1,cfg_train.total_epoch,  running_loss/len(dataloader_train)))
            f.write('\n')

        if (epoch+1) > 25 and (epoch+1)%5 == 0:
            model_path = cfg_train.savepath + 'salient_rgb_model_sharewgt_' + \
                str(epoch+1) + '.pth'
            torch.save(model.state_dict(), model_path)



if __name__ == '__main__':
    file = open("log.txt", 'w').close()  # remove content in log.txt
    Train(Dataset, edgenet)

    print("That's it! Done...")

#==================================================================================#
