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
import random
import sys

class Dataset():
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, index):
        im = Image.open(self.data[index][0])
        im = self.transform(im)
        return im, self.data[index][1]
    
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
    train_image = sorted(glob.glob(os.path.join(sys.argv[1], '*.jpg')))
    train_label = pd.read_csv(sys.argv[2])
    train_label = train_label.iloc[:,1].values.tolist()
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)

    train_label = train_data[:round(len(train_data)*0.95)] 
    valid_label = train_data[round(len(train_data)*0.95):]
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_set = Dataset(train_data, transform)
    valid_set = Dataset(valid_label, transform)
    train_loader = Data.DataLoader(train_set, batch_size = 64, shuffle = True)
    valid_loader = Data.DataLoader(valid_set, batch_size = 64, shuffle = False)
    
    model = Net()
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    EPOCH = 100
    acc_train_his = []
    loss_train_his = []
    acc_valid_his = []
    loss_valid_his = []

    for epoch in range(EPOCH):
        model.train()
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        acc_train_his.append(np.mean(train_acc))
        loss_train_his.append(np.mean(train_loss))


        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
            acc_valid_his.append(np.mean(valid_acc))
            loss_valid_his.append(np.mean(valid_loss))

        if np.mean(train_acc) > 0.9:
            checkpoint_path = 'model_{}.pth'.format(epoch+1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)
    