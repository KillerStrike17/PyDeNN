from DeNN.base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class Quiz_conv(BaseModel):
    def __init__(self):
        DROPOUT_VALUE = 0.2
        
        super(Quiz_conv, self).__init__()

        self.conv1 = nn.Sequential(
          nn.Conv2d(3, 32, 1,padding = 1, bias=False), 
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(DROPOUT_VALUE)
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding = 1, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(DROPOUT_VALUE)
        )
        self.conv3 = nn.Sequential(
          nn.Conv2d(32, 32, 3,padding = 1, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(32),
          nn.Dropout(DROPOUT_VALUE),
        )

        self.pool1 = nn.Sequential(
          nn.MaxPool2d(2, 2), 

        ) 
        self.conv4 = nn.Sequential(
          nn.Conv2d(32, 64, 1, bias=False), 
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(DROPOUT_VALUE)
        )

        self.conv5 = nn.Sequential(
          nn.Conv2d(64, 64, 3,padding = 1, bias=False),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(DROPOUT_VALUE)
        )
          
        self.conv6 = nn.Sequential(
          nn.Conv2d(64, 64, 3,padding = 1, bias=False), 
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(DROPOUT_VALUE)
        )

        self.pool2 = nn.Sequential(
          nn.MaxPool2d(2, 2), 
        )

        self.conv7 = nn.Sequential(
          nn.Conv2d(64, 128, 1,padding = 1, bias=False), 
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(DROPOUT_VALUE)
        )       

        self.conv8 = nn.Sequential(
          nn.Conv2d(128, 128, 3,padding = 1, bias=False), 
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(DROPOUT_VALUE)
        )   
        self.conv9 = nn.Sequential(
          nn.Conv2d(128, 128, 3,padding = 1, bias=False), 
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(DROPOUT_VALUE)
        ) 

        self.conv10 = nn.Sequential(
          nn.Conv2d(128, 128, 3,padding = 1, bias=False), 
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(DROPOUT_VALUE)
        )   

        self.gap = nn.Sequential(  
          nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.fc = nn.Sequential(
            nn.Conv2d(128, 10, 1)
        )
    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1 + x2)
        x4 = self.pool1(x1 + x2 + x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x4)
        x6 = self.conv6(x4 + x5)
        x7 = self.conv6(x4 + x5 +x6)
        x8 = self.pool2(x5 + x6 + x7)
        x8 = self.conv7(x8)
        x9 = self.conv8(x8)
        x10 = self.conv9(x8 + x9)
        x11 = self.conv10(x8 + x9 + x10)
        x12 = self.gap(x11)
        x13 = self.fc(x12)
        x14 = x13.view(-1, 10)

        return x14