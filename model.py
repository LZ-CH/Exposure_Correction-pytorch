'''
@Autuor: LZ-CH
@Contact: 2443976970@qq.com
'''

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
class cnnblock(nn.Module):
    def __init__(self,in_channle,out_channle):
        super(cnnblock,self).__init__()
        self.cnn_conv1=nn.Conv2d(in_channle,out_channle,3,1,1)
        self.ac1=nn.LeakyReLU(inplace = True)

        self.cnn_conv2=nn.Conv2d(out_channle,out_channle,3,1,1)
        self.ac2=nn.LeakyReLU(inplace = True)
    
    def forward(self,x):
        x=self.cnn_conv1(x)
        x=self.ac1(x)
        x=self.cnn_conv2(x)
        x=self.ac2(x)
        return x
class Upsample(nn.Module):
    """Upscaling"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels,out_channels,3,1,1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ac=nn.LeakyReLU(inplace = True)

    def forward(self, x, shape1, shape2):
        x = self.up(x)
        # input is CHW
        diffY = shape1 - x.shape[2]
        diffX = shape2 - x.shape[3]
        if self.bilinear:
            x = self.conv(x)
        x = self.ac(x)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        return x


class SubNet_3layers(nn.Module):
    def __init__(self, firstoutputchannl = 64):
        super(SubNet_3layers,self).__init__()
        self.outputchannl = 3
        self.maxpool=nn.MaxPool2d(2)
        self.block1=cnnblock(3,firstoutputchannl)
        self.block2=cnnblock(firstoutputchannl,2*firstoutputchannl)        
        self.block3=cnnblock(2*firstoutputchannl,4*firstoutputchannl)
        self.block4=cnnblock(4*firstoutputchannl,8*firstoutputchannl)
        self.up1=Upsample(8*firstoutputchannl,4*firstoutputchannl)
        self.block5=cnnblock(8*firstoutputchannl,4*firstoutputchannl)
        self.up2=Upsample(4*firstoutputchannl,2*firstoutputchannl)
        self.block6=cnnblock(4*firstoutputchannl,2*firstoutputchannl)
        self.up3=Upsample(2*firstoutputchannl,firstoutputchannl)
        self.block7=cnnblock(2*firstoutputchannl,firstoutputchannl)
        self.finalconv=nn.Conv2d(firstoutputchannl,self.outputchannl,1,1,0)

    def forward(self,x):
        out1=self.block1(x)
        out2=self.block2(self.maxpool(out1))
        out3=self.block3(self.maxpool(out2))
        out4=self.block4(self.maxpool(out3))
        in5=torch.cat([self.up1(out4,out3.shape[2],out3.shape[3]),out3],1)
        out5=self.block5(in5)
        in6=torch.cat([self.up2(out5,out2.shape[2],out2.shape[3]),out2],1)
        out6=self.block6(in6)
        in7=torch.cat([self.up3(out6,out1.shape[2],out1.shape[3]),out1],1)
        out7=self.block7(in7)
        predict= self.finalconv(out7)
        return predict

class SubNet_4layers(nn.Module):
    def __init__(self, firstoutputchannl = 64):
        super(SubNet_4layers,self).__init__()
        self.outputchannl = 3
        self.block1=cnnblock(3,firstoutputchannl)
        self.maxpool=nn.MaxPool2d(2)
        self.block2=cnnblock(firstoutputchannl,2*firstoutputchannl)        
        self.block3=cnnblock(2*firstoutputchannl,4*firstoutputchannl)
        self.block4=cnnblock(4*firstoutputchannl,8*firstoutputchannl)
        self.block5=cnnblock(8*firstoutputchannl,16*firstoutputchannl)

        self.up1=Upsample(16*firstoutputchannl,8*firstoutputchannl)
        self.block6=cnnblock(16*firstoutputchannl,8*firstoutputchannl)

        self.up2=Upsample(8*firstoutputchannl,4*firstoutputchannl)
        self.block7=cnnblock(8*firstoutputchannl,4*firstoutputchannl)

        self.up3=Upsample(4*firstoutputchannl,2*firstoutputchannl)
        self.block8=cnnblock(4*firstoutputchannl,2*firstoutputchannl)

        self.up4=Upsample(2*firstoutputchannl,firstoutputchannl)
        self.block9=cnnblock(2*firstoutputchannl,firstoutputchannl)
        self.finalconv=nn.Conv2d(firstoutputchannl,self.outputchannl,1,1,0)

    def forward(self,x):
        out1=self.block1(x)
        out2=self.block2(self.maxpool(out1))
        out3=self.block3(self.maxpool(out2))
        out4=self.block4(self.maxpool(out3))
        out5=self.block5(self.maxpool(out4))
        in6=torch.cat([self.up1(out5,out4.shape[2],out4.shape[3]),out4],1)
        out6=self.block6(in6)
        in7=torch.cat([self.up2(out6,out3.shape[2],out3.shape[3]),out3],1)
        out7=self.block7(in7)
        in8=torch.cat([self.up3(out7,out2.shape[2],out2.shape[3]),out2],1)
        out8=self.block8(in8)
        in9=torch.cat([self.up4(out8,out1.shape[2],out1.shape[3]),out1],1)
        out9=self.block9(in9)
        predict=self.finalconv(out9)
        return predict

class MSPEC_Net(nn.Module):
    def __init__ (self):
        super (MSPEC_Net,self).__init__()
        self.subnet1 = SubNet_4layers(24)
        self.subnet2 = SubNet_3layers(24)
        self.subnet3 = SubNet_3layers(24)
        self.subnet4 = SubNet_3layers(16)
        self.up1 = Upsample(3,3)
        self.up2 = Upsample(3,3)
        self.up3 = Upsample(3,3)
    def forward(self,L_list):
        y1_0 = self.subnet1(L_list[0])
        y1_1 = self.up1(y1_0,L_list[1].shape[2],L_list[1].shape[3])
        y2_0 = self.subnet2(y1_1 + L_list[1]) + y1_1
        y2_1 = self.up2(y2_0,L_list[2].shape[2],L_list[2].shape[3])
        y3_0 = self.subnet3(y2_1 + L_list[2]) + y2_1
        y3_1 = self.up3(y3_0,L_list[3].shape[2],L_list[3].shape[3])
        y4 = self.subnet4(y3_1 + L_list[3]) + y3_1
        Y_list = [y1_1,y2_1,y3_1,y4]
        return Y_list
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3,8,4,2,1)
        self.ac1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(8,16,4,2,1)
        self.bn2 = nn.BatchNorm2d(16)
        self.ac2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(16,32,4,2,1)
        self.ac3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(32,64,4,2,1)
        self.ac4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(64,128,4,2,1)
        self.ac5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(128,128,4,2,1)
        self.ac6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(128,256,4,2,1)
        self.ac7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(256,1,2,2,0)
    def forward(self,x):
        if x.shape[2]!=256 and x.shape[3]!=256:
            x = F.interpolate(x,(256,256),mode='bilinear',align_corners=True)
        y = self.ac1(self.conv1(x))
        y = self.ac2(self.bn2(self.conv2(y)))
        y = self.ac3(self.conv3(y))
        y = self.ac4(self.conv4(y))
        y = self.ac5(self.conv5(y))
        y = self.ac6(self.conv6(y))
        y = self.ac7(self.conv7(y))
        y = self.conv8(y)
        return y
if __name__ == '__main__':
    a = torch.randn(2, 3, 128, 128)
    dec = Discriminator()
    y =dec(a)
    print(y.shape)