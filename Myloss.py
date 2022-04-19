'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np

class Pyr_Loss (nn.Module):
    def __init__(self,weight=1.0):
        super(Pyr_Loss, self).__init__()
        self.weight =weight
        self.criterion = nn.L1Loss(reduction='sum')
    def forward(self,Y_list, T_list):
        n = len(Y_list)
        loss = 0
        for m in range(0,n-1):
            loss += self.weight*(2**(n-m-2))*self.criterion(Y_list[m],F.interpolate(T_list[m],(Y_list[m].shape[2],Y_list[m].shape[3]),mode='bilinear',align_corners=True))/Y_list[m].shape[0]
        return loss

class Rec_Loss(nn.Module):
    def __init__(self,weight=1):
        super(Rec_Loss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction='sum')
    def forward(self,Y_list, T_list):
        loss = self.weight * self.criterion(Y_list[-1],T_list[-1])/Y_list[-1].shape[0]
        return loss

class Adv_loss(nn.Module):
    def __init__(self,size =256 ,weight=1.0):
        super(Adv_loss,self).__init__()
        self.weight = weight
        self.size = size
    def forward(self,P_Y):
        loss = -self.weight * 12 *self.size*self.size*torch.mean(torch.log(torch.sigmoid(P_Y)+1e-9))
        return loss

class My_loss(nn.Module):
    def __init__(self,size =256,Pyr_weight = 1.0,Rec_weight = 1.0, Adv_weight = 1.0):
        super(My_loss,self).__init__()
        self.pyr_loss = Pyr_Loss(Pyr_weight)
        self.rec_loss = Rec_Loss(Rec_weight)
        self.adv_loss = Adv_loss(size,Adv_weight)
    def forward(self,Y_list,T_list,P_Y = None,withoutadvloss = False):
        pyrloss =self.pyr_loss(Y_list, T_list)
        recloss =self.rec_loss(Y_list,T_list)
        if withoutadvloss:
            myloss = pyrloss + recloss
            return recloss,pyrloss,myloss
        else:
            advloss =self.adv_loss(P_Y)
            myloss = pyrloss + recloss + advloss
            return recloss, pyrloss, advloss, myloss


class D_loss(nn.Module):
    def __init__(self):
        super(D_loss,self).__init__()
    def forward(self,P_Y,P_T):
        loss = -torch.mean(torch.log(torch.sigmoid(P_T) + 1e-9)) - torch.mean(torch.log(1 - torch.sigmoid(P_Y) + 1e-9))
        return loss

if __name__ =='__main__':
    a = torch.rand([2,3,32,32]).float()
    p = torch.rand([2,1]).float()
    alist = [a]
    total_loss = My_loss()
    t = total_loss(alist,alist,p)
    print(t)
        