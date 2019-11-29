import os
from PIL import Image
import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import glob
import sys

def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

class test_Dataset():
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, index):
        im = Image.open(self.data[index])
        im = self.transform(im)
        return im, 0
    
    def __len__(self):
        return len(self.data)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),     
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),            
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        #image size (48,48)
        x = self.conv1(x) #(24,24)
        x = self.conv2(x) #(12,12)
        x = self.conv3(x) #(6,6)
        x = self.conv4(x) #(3,3)
        x = x.view(-1, 3*3*128)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('model_70.pth'))
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    with torch.no_grad():
        test_image = sorted(glob.glob(os.path.join(sys.argv[1], '*.jpg')))
        test_set = test_Dataset(test_image, transform)
        test_loader = Data.DataLoader(test_set, batch_size = 128, shuffle = False)
        test_preds = get_all_preds(model, test_loader)
        predict = torch.max(test_preds, 1)[1]
    
    end = pd.DataFrame({'id': range(0,len(test_image)), 'label': predict})
    end.to_csv(sys.argv[2], index = False)
    
    
    