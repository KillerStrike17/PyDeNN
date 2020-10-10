from DeNN.base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class David_net(BaseModel):
    def __init__(self):
        super(David_net, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3,padding = 1), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.x = nn.Sequential(    
            nn.Conv2d(64,128,3,padding = 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
        )
        self.r1 = nn.Sequential(
            nn.Conv2d(128, 128, 3,padding = 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3,padding = 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # self.layer1 = self.x + self.r1
        self.layer2 =  nn.Sequential(
            nn.Conv2d(128, 256, 3,padding = 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.x2 = nn.Sequential(
            nn.Conv2d( 256, 512,3,padding = 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.r2 = nn.Sequential(
            nn.Conv2d(512, 512, 3,padding = 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3,padding = 1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # self.layer3 = self.x2 + self.r2
        self.pool  = nn.Sequential(
            nn.MaxPool2d(4,4)
        )
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1, stride=1, bias=False)
        )
        self.stmax =nn.Softmax(dim=1)
    
    def forward(self,x):
        out = self.prep(x)
        x1 = self.x1(x)
        r1 = self.r1(x1)
        l1 = x1+r1
        l2 = self.l2(l1)
        x2 = self.x2(l2)
        r2 = self.r1(x2)
        l2 = x2+r2
        p1 = self.pool(l2)
        fc = self.fc(p1)
        stmax = self.stmax(fc)
        out = stmax.view(stmax.size(0), -1)
        return out