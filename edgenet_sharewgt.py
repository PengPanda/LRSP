import torch
from torch import nn
import torch.nn.functional as F
from long_range_selective_pooling import LRSpool2d

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#==================================================================================#

class edgenet(nn.Module):

    def __init__(self, cfg):
        super(edgenet, self).__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(3, 94, kernel_size=5, stride=2,padding=2) # in_chnl = 1 or 3
        self.bn1 = nn.BatchNorm2d(94)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(94, 128, kernel_size=5, stride=2,padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv11 = nn.Conv2d(256,128, kernel_size=3, stride=1,padding=1)  
        self.bn11 = nn.BatchNorm2d(128)		
        self.lrsp11 = LRSpool2d(in_channels=128,num_ori=16)  # 20210403 test

        # self.conv22 = nn.Conv2d(256,128, kernel_size=3, stride=1,padding=1)  
        # self.bn22 = nn.BatchNorm2d(128)		
        self.lrsp22 = LRSpool2d(in_channels=128,num_ori=16)  # 20210403 test

        self.conv4 = nn.Conv2d(256,128, kernel_size=3, stride=1,padding=1)  
        self.bn4 = nn.BatchNorm2d(128)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv5 = nn.Conv2d(128, 94, kernel_size=3, stride=1,padding=1) 
        self.bn5 = nn.BatchNorm2d(94)
    
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv6 = nn.Conv2d(94, 1, kernel_size=3, stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(1)

        # Initialize convolutions' parameters
        self.initialize() 

    def initialize(self):
            if self.cfg.snapshot:
                self.load_state_dict(torch.load(self.cfg.snapshot))            
            else:
                for c in self.children():
                    if isinstance(c, nn.Conv2d):
                        nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity='relu')
                        # nn.init.xavier_uniform_(c.weight)
                        nn.init.constant_(c.bias, 0.)

    def forward(self, image, aij, sij):

        out =self.conv1(image)
        out = F.relu(self.bn1(out))
        out = self.pool1(out)   

        out = self.conv2(out)
        out = F.relu(self.bn2(out)) 
        out = self.pool2(out) 

        out = self.conv3(out)
        out_temp0 = out;
        out0 = F.relu(self.bn3(out))

        out1 = self.conv11(out0)
        out1 = F.relu(self.bn11(out1))
        out1 = self.lrsp11(out1,aij)  # long-range pooling - shape

        out2 = self.conv11(out0)
        out2 = F.relu(self.bn11(out2))
        out2 = self.lrsp22(out2,sij)  # long-range pooling - shape

        out = torch.cat([out1, out2], 1)
        out = self.conv4(out)
        out = F.relu(self.bn4(out))
        # out_temp1 = out;

        out = self.upsample1(out)
        out = self.conv5(out)
        out = F.relu(self.bn5(out))

        out = self.upsample2(out)
        out = self.conv6(out) 
        out = self.bn6(out)

        # out = torch.sigmoid(out) 

        return out

# ========================================================== #

# # test...
if __name__=='__main__':
    
    import torch
    img = torch.randn(2,1,256, 256)
    aij = torch.randn(2, 16, 256, 256)
    sij = torch.randn(2, 16, 256, 256)
    net = edgenet()
    out = net(img,aij,sij)
    print(out.size())

    # net = edgenet().to(device)
    # # summary(net, [(1, 256, 256),(16, 256, 256),(48,16,16)])
    # print(net)