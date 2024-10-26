import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights
from models.HRNet import hrnet32
import math

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
                              

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        # self.eca = ECABlock(planes)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # out = self.eca(out)
        out += identity
        out = self.relu(out)

        return out

    
 

class Fpn(nn.Module):
    def __init__(self):
        super(Fpn,self).__init__()
        out = [256,384,448,480]
        self.conv1 = nn.Sequential(nn.Conv2d(480, 256, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True)
                                       )
        self.conv2 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(384),
                                       nn.ReLU(inplace=True)
                                       )
        self.conv3 = nn.Sequential(nn.Conv2d(448, 448, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(448),
                                       nn.ReLU(inplace=True)
                                       )
        
        self.smooth1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
                                     )
        self.smooth2 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(96),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
                                     )
        self.smooth3 = nn.Sequential(nn.Conv2d(224, 224, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(224),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(224 ,224, kernel_size=3, stride=2, padding=1)
                                     )
    #    self.smooth1 = nn.Sequential(
    #                                  nn.BatchNorm2d(32),
    #                                  nn.ReLU(inplace=True),
    #                                  nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
    #                                  )
    #    self.smooth2 = nn.Sequential(
    #                                  nn.BatchNorm2d(96),
    #                                  nn.ReLU(inplace=True),
    #                                  nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
    #                                  )
    #    self.smooth3 = nn.Sequential(
    #                                  nn.BatchNorm2d(224),
    #                                  nn.ReLU(inplace=True),
    #                                  nn.Conv2d(224 ,224, kernel_size=3, stride=2, padding=1)
    #                                  )

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = [block(inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x1,x2,x3,x4):
        # print(x1.size(),x2.size(),x3.size(),x4.size())
        p1 = self.smooth1(x1)
        p1 = torch.cat([p1,x2],1)
        p2 =self.smooth2(p1)
        p2 =torch.cat([p2,x3],1)
        p3 = self.smooth3(p2)
        p3 = torch.cat([p3,x4],1)
          
        s4 = self.conv1(p3)
        x4 = F.interpolate(s4, size=32, mode='bilinear', align_corners=True)
        s1 =torch.cat([x4,x3],1)
        x3 =self.conv2(s1)
        x3 = F.interpolate(x3, size=64, mode='bilinear', align_corners=True)
        s2 =torch.cat([x3,x2],1)
        x2 = self.conv3(s2)
        x2 =  F.interpolate(x2, size=128, mode='bilinear', align_corners=True)
        s3 =torch.cat([x2,x1],1) 
       
        return s3


class ECABlock (nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        v = self.avg_pool(x)
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(v)
        return x * v
    


           
class BasicNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(BasicNet, self).__init__()
        self.hrnet = hrnet32(pretrained=True)
        self.classifier1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, num_classes, kernel_size=1))
        self.classifier2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, num_classes, kernel_size=1))
        self.fpn =Fpn()
        self.resCD = self._make_layer(ResBlock, 544, 128, 6, stride=1)
        self.eca = ECABlock(544)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1))
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=480,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0)
        )  
        initialize_weights(self.fpn, self.resCD, self.classifierCD, self.classifier1, self.classifier2)
       
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def base_forward(self, x):
        x,x1,x2,x3,x4 = self.hrnet(x)
        s1 = self.fpn(x1,x2,x3,x4)
        
        s1= self.last_layer(s1) 
             
        return s1, x1
    
    def CD_forward(self,x1,x2,d1):
        x= torch.cat([x1,x2,d1],1)
        x= self.eca(x)
        x = self.resCD(x)
        
        change = self.classifierCD(x)
        
        return change
    
    def forward(self, x1, x2):
        x_size = x1.size()
        s1,x1= self.base_forward(x1)
        s2,x2 = self.base_forward(x2)
        
        d1 = abs(x1-x2)#128
       
        change = self.CD_forward(s1,s2,d1)   
    
        out1 = self.classifier1(s1)
        out2 = self.classifier2(s2)

        out1 =F.upsample(out1, x_size[2:], mode='bilinear')
        out2 =F.upsample(out2, x_size[2:], mode='bilinear')
        change = F.upsample(change, x_size[2:], mode='bilinear')
       
        return change,out1,out2


# from thop import profile


# net = BasicNet().cuda()
# x1 = torch.Tensor(1, 3, 512, 512).cuda()
# x2 = torch.Tensor(1, 3, 512, 512).cuda()
# change, out1, out2 = net(x1, x2)
# print(change.shape)

# flops, params = profile(net, (x1, x2))
# print("FLOPs: ", flops / 1e9, "params: ", params / 1e6)
