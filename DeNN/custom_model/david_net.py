from DeNN.base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class Resnet_block(BaseModel):
    def __init__(self,kernels, channels):
        super(Resnet_block,self).__init__()
        self.c1 = nn.Conv2d(kernels, channels, 3,padding = 1,bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.re = nn.ReLU()
        self.c2 = nn.Conv2d(channels, channels, 3,padding = 1,bias=False)
        self.b2 = nn.BatchNorm2d(channels)
        self.re2 = nn.ReLU()
        self.residue = nn.Sequential()
    
    def forward(self,x):
        out = self.re(self.b1(self.c1(x)))
        out = self.re2(self.b2(self.c2(out)))
        out = out + self.residue(out)
        return out

class Conv_block(BaseModel):
    def __init__(self, kernels, channels):
        super(Conv_block,self).__init__()
        self.c1 = nn.Conv2d(kernels, channels, 3,padding = 1,bias=False)
        self.mp = nn.MaxPool2d(2,2)
        self.b1 = nn.BatchNorm2d(channels)
        self.re = nn.ReLU()
        self.resnet = Resnet_block(kernels,channels)
    
    def forward(self,x):
        out = self.re(self.b1(self.mp(self.c1(x))))
        res = self.resnet(out)
        return out + res



class David_net(BaseModel):
    def __init__(self):
        super(David_net, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3,padding = 1,bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.l1 = Conv_block(64,128)
        self.l2 =  nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1,bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.l3 = Conv_block(256,512)
        self.mp = nn.MaxPool2d(4,2)
        self.fc = nn.Conv2d(512, 10, 1, bias=False)

    
    def forward(self,x):
        out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.mp(out)
        out = self.fc(out)
        out = out.view(out.size(0), -1)
        return out