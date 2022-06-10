import torch
from torch import nn
from torch.nn import functional as F

class LRSpool2d(nn.Module):
    def __init__(self, in_channels, num_ori=8):

        super(LRSpool2d, self).__init__()

        self.in_channels = in_channels
        self.num_ori = num_ori
        self.g = nn.Conv2d(in_channels=self.in_channels*self.num_ori, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)

        # self.bn = nn.BatchNorm2d(self.in_channels)
        self.layernorm = nn.LayerNorm([16, 16])

    def forward(self, x, aij):
        """
        :param x: (b, c, h, w)
        :param aij: (b, n, h*w, h*w)
        :return:
        """
        batch_size = x.size(0)   # x -> [b,c,h,w]
 
        t_x = x.view(batch_size, self.in_channels, -1)  # [b,c,hw]
        t_x = t_x.permute(0, 2, 1)             # [b,hw,c]

        t_x = t_x.unsqueeze(dim=1)             # [b,1,hw,c]
        y = torch.matmul(aij, t_x)             # [b,n,hw,hw] x [b,1,hw,c] = [b,n,hw,c] 
        
        y = y.permute(0, 1, 3, 2).contiguous() # [b,n,c,hw]
        y = y.view(batch_size, self.in_channels*self.num_ori, *x.size()[2:]) # y -> [b,n*c,h,w]

        # g_y = self.g(y)                        # g_y -> [b,c,h,w]
        # z = self.bn(g_y)
                
        y = self.layernorm(y)
        z = self.g(y)  

        return z


if __name__ == '__main__':

    import torch

    img = torch.zeros(2, 16, 20, 20)
    Aij = torch.zeros(2, 8, 400, 400)
    net = LRSpool2d(16,8)
    out = net(img,Aij)
    
    print(out.size())
