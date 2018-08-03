import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class VggNet(nn.Module):
    
    def __init__(self):
        super(VggNet, self).__init__()
        self.pool1=nn.MaxPool2d(2,2)
        self.pool2=nn.MaxPool2d(2,2)
        self.pool3=nn.MaxPool2d(2,2)
        self.pool4=nn.MaxPool2d(2,2)
        self.pool5=nn.MaxPool2d(2,2)
        self.conv1=nn.Conv2d(3,64,3,padding=1)

        self.conv2=nn.Conv2d(64,128,3,padding=1)

        self.conv3=nn.Conv2d(128,256,3,padding=1)
        self.conv4=nn.Conv2d(256,256,3,padding=1)

        self.conv5=nn.Conv2d(256,512,3,padding=1)
   
        self.conv6=nn.Conv2d(512,512,3,padding=1)
        self.conv7=nn.Conv2d(512,512,3,padding=1)

        self.conv8=nn.Conv2d(512,512,3,padding=1)
        self.fc1=nn.Linear(512,512)
        self.fc2=nn.Linear(512,512)
        self.fc3=nn.Linear(512,10)
        self.Dropout=nn.Dropout(0.5)
    def forward(self,x):

        x=self.pool1(F.relu(self.conv1(x)))

        x=self.pool2(F.relu(self.conv2(x)))
        x=F.relu(self.conv3(x))

        x=self.pool3(F.relu(self.conv4(x)))
 
        x=F.relu(self.conv5(x))
        x=self.pool4(F.relu(self.conv6(x)))
        x=F.relu(self.conv7(x))

        x=self.pool5(F.relu(self.conv8(x)))
        x=x.view(-1,self.num_flat_features(x))
        x=self.Dropout(x)
        x=F.relu(self.fc1(x))
        x=self.Dropout(x)
        x=F.relu(self.fc2(x))
        x=self.Dropout(x)
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        num_features=1
        size=x.size()[1:]
        for s in size:
            num_features*=s
        return num_features
