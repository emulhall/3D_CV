import logging
from typing import List
import torch
import torch.nn as nn
import torchvision


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleDepthNet(nn.Module):
    def __init__(self):
        super(SimpleDepthNet, self).__init__()

        self.channels = [3, 8, 16, 32]

        self.encoder = nn.Sequential(
                # layer 1
                nn.Conv2d(self.channels[0], self.channels[1], 3, 1, 1),
                nn.BatchNorm2d(self.channels[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=4, stride=4),
                # layer 2
                nn.Conv2d(self.channels[1], self.channels[2], 3, 1, 1),
                nn.BatchNorm2d(self.channels[2]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=4, stride=4),
                # layer 3
                nn.Conv2d(self.channels[2], self.channels[3], 3, 1, 1),
                nn.BatchNorm2d(self.channels[3]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.decoder = nn.Sequential(
                nn.Conv2d(self.channels[3], self.channels[2], 3, 1, 1),                
                nn.ReLU(inplace=True),
                nn.Upsample(size=(15, 20), mode='bilinear', align_corners=False),

                nn.Conv2d(self.channels[2], self.channels[1], 3, 1, 1),                
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),

                nn.Conv2d(self.channels[1], 1, 1),                
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
            )

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)        
        return y


class ExtendedFPNDepth(nn.Module):
    def __init__(self):
        super(ExtendedFPNDepth,self).__init__()

        self.coarse_channels=[3,96,256,384,13824,4096,4524]
        self.fine_channels=[3,63,64,1]

        #Establish the coarse network
        self.coarse_conv1=nn.Conv2d(self.coarse_channels[0],self.coarse_channels[1], kernel_size=11, stride=4)
        self.coarse_conv2=nn.Conv2d(self.coarse_channels[1], self.coarse_channels[2], kernel_size=5, padding=2)
        self.coarse_conv3=nn.Conv2d(self.coarse_channels[2], self.coarse_channels[3], kernel_size=3, padding=1)
        self.coarse_conv4=nn.Conv2d(self.coarse_channels[3], self.coarse_channels[3], kernel_size=3, padding=1)
        self.coarse_conv5=nn.Conv2d(self.coarse_channels[3], self.coarse_channels[2], kernel_size=3, stride=2)
        self.coarse_fc1 = nn.Linear(self.coarse_channels[4],self.coarse_channels[5])
        self.coarse_fc2 = nn.Linear(self.coarse_channels[5],self.coarse_channels[6])

        #Establish the fine network
        self.fine_conv1=nn.Conv2d(self.fine_channels[0],self.fine_channels[1], kernel_size=9, stride=2)
        self.fine_conv2=nn.Conv2d(self.fine_channels[2],self.fine_channels[2],kernel_size=5, padding=2)
        self.fine_conv3=nn.Conv2d(self.fine_channels[2],self.fine_channels[3], kernel_size=5, padding=2)

        #Establish pooling, and relu
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d()
        self.ReLU = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(size=(240,320), mode='bilinear', align_corners=False)


    def forward(self, x):
        # Design network
        ## Coarse network
        #Layer 1
        h = self.coarse_conv1(x)
        h = self.ReLU(h)
        h = self.pool(h) #B x 96 x 29 x 39
        #Layer 2
        h = self.coarse_conv2(h)
        h = self.ReLU(h)
        h = self.pool(h) #B x 256 x 14 x 19
        #Layer 3
        h = self.coarse_conv3(h)
        h = self.ReLU(h) #B x 384 x 14 x 19
        #Layer 4
        h = self.coarse_conv4(h)
        h = self.ReLU(h) #B x 384 x 14 x 19
        #Layer 5
        h = self.coarse_conv5(h)
        h = self.ReLU(h) #B x 256 x 6 x 9
        #Reshape to be fully connected
        h = h.view(h.shape[0],-1) #B x 13824
        #Layer 6
        h = self.coarse_fc1(h)
        h = self.ReLU(h)
        h = self.dropout(h) #B x 4096
        #Layer 7
        h = self.coarse_fc2(h) #B x 4524
        #Reshape to match the size of the fine network
        h = h.view(x.shape[0],1,58,78)

        ##Fine network
        #Layer 1
        x = self.fine_conv1(x)
        x = self.ReLU(x)
        x = self.pool(x) #B x 63 x 58 x 78
        #Add in the coarse network output
        x = torch.cat((x,h),1) #B x 64 x 58 x 78
        #Layer 2
        x = self.fine_conv2(x)
        x = self.ReLU(x) #B x 64 x 58 x 78
        #Layer 3
        x = self.fine_conv3(x)
        x = self.ReLU(x) #B x 1 x 58 x 78
        y = self.upsample(x)
        return y
