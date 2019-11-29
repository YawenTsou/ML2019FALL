import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import sys

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 3, 2, 1), # 10, 16, 16
            nn.Conv2d(256, 128, 3, 2, 1), # 20, 8, 8
            nn.Conv2d(128, 80, 3, 2, 1),
            nn.Conv2d(80, 40, 3, 2, 1), # 5, 4, 4
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 80, 2, 2), # 20, 8, 8
            nn.ConvTranspose2d(80, 128, 2, 2),
            nn.ConvTranspose2d(128, 256, 2, 2), # 10, 16, 16
            nn.ConvTranspose2d(256, 3, 2, 2), # 3, 32, 32
            nn.Tanh()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return encoded, decoded

    
    
if __name__ == '__main__':
    train_x = np.load(sys.argv[1])
    train_x = np.transpose(train_x, (0, 3, 1, 2)) / 255. * 2 -1
    train_x = torch.Tensor(train_x)
    
    use_gpu = torch.cuda.is_available()
    autoencoder = Autoencoder()

    if use_gpu:
        autoencoder.cuda()
        train_x = train_x.cuda()

    train_dataloader = DataLoader(train_x, batch_size=32, shuffle=True)
    
    criteria = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    EPOCH = 20

    for epoch in range(EPOCH):
        cumulate_loss = 0
        for x in train_dataloader:
            latent, reconstruct = autoencoder(x)
            loss = criteria(reconstruct, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulate_loss += loss.item() * x.shape[0]
        print(f'Epoch { "%03d" % epoch }: Loss : { "%.5f" % (cumulate_loss / train_x.shape[0])}')


    checkpoint_path = 'model_{}.pth'.format(epoch+1) 
    torch.save(autoencoder.state_dict(), checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
    
