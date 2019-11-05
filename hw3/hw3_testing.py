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


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('model_70.pth'))
    model.eval()
    
    with torch.no_grad():
        test_image = sorted(glob.glob(os.path.join(sys.argv[1], '*.jpg')))
        test_set = test_Dataset(test_image, transform)
        test_loader = Data.DataLoader(test_set, batch_size = 128, shuffle = False)
        test_preds = get_all_preds(model, test_loader)
        predict = torch.max(test_preds, 1)[1]
    
    end = pd.DataFrame({'id': range(0,len(test_image)), 'label': predict})
    end.to_csv(sys.argv[2], index = False)
    
    
    