from DeNN.base import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class Adv_conv(BaseModel):
    def __init__(self):
        DROPOUT_VALUE = 0.2
        super(Adv_conv, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.conv1 = nn.Sequential(
          # RF 3x3
          nn.Conv2d(3, 64, 3,padding = 1), #32
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(DROPOUT_VALUE),

          # RF 5x5
          nn.Conv2d(64, 64, 3,padding = 1), #30
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(DROPOUT_VALUE),
          
          # RF 7x7
          nn.Conv2d(64, 64, 3,padding = 1), # 28
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Dropout(DROPOUT_VALUE),
        )

        self.trans1 = nn.Sequential(
          # RF 7x7 
          nn.Conv2d(64, 32, 1,padding = 1), # 28

          # RF 14x14
          nn.MaxPool2d(2, 2), # 14

        ) 
        self.conv2 = nn.Sequential(
          # RF 3x3
          nn.Conv2d(32, 128, 3), #32
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(DROPOUT_VALUE),

          # RF 5x5
          nn.Conv2d(128, 128, 3), #30
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(DROPOUT_VALUE),
          
          # RF 7x7
          nn.Conv2d(128, 128, 3), # 28
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Dropout(DROPOUT_VALUE),
        )
        self.trans2 = nn.Sequential(
          # RF 7x7 
          nn.Conv2d(128, 64, 1,padding = 1), # 28

          # RF 14x14
          nn.MaxPool2d(2, 2), # 14
        ) 

        self.conv3 = nn.Sequential(
          # RF 3x3
          # Depthwise Seperable Convolution
          nn.Conv2d(64, 64, kernel_size = 3,padding=1, groups=64), #32
          nn.Conv2d(64, 256, kernel_size=1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Dropout(DROPOUT_VALUE),

          # RF 5x5
          nn.Conv2d(256, 256, 3,dilation = 1), #30
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Dropout(DROPOUT_VALUE),
          
          # RF 7x7
          nn.Conv2d(256, 256, 3), # 28
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Dropout(DROPOUT_VALUE),
        )

        self.trans3 = nn.Sequential(
          # RF 7x7 
          nn.Conv2d(256, 128, 1,padding = 1), # 28

          # RF 14x14
          nn.MaxPool2d(2, 2), # 14

        )

        self.gap = nn.Sequential(  
           
          nn.AvgPool2d(7,7),
          # RF 18x18
          nn.Conv2d(20, 10, 1),
        )
    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.conv1(x)
        x = self.trans1(x)
        x = self.conv2(x)
        x = self.trans2(x)
        x = self.conv3(x)
        x = self.trans3(x)

        return x
