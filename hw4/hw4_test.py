import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
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
    
    test_dataloader = DataLoader(train_x, batch_size=64, shuffle=False)
    autoencoder = Autoencoder()
    # model = Resnet18()
    autoencoder.load_state_dict(torch.load('model_20_.pth'))
    
    latents = []
    reconstructs = []
    for x in test_dataloader:
        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())
        
        
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)
    
    latents = TSNE(n_components=2).fit_transform(latents)
    
    result = KMeans(n_clusters=6).fit(latents).labels_
    result1 = AgglomerativeClustering(n_clusters=6).fit(latents).labels_
    result2 = GaussianMixture(n_components=6).fit_predict(latents)
    result3 = AgglomerativeClustering(n_clusters=6, linkage='complete').fit(latents).labels_
    result4 = AgglomerativeClustering(n_clusters=6, linkage='average').fit(latents).labels_
    
    seed = [0, 1, 2, 3, 4, 5]
    
    cluster = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    for i in seed:
        cluster.append(result[i])
        cluster1.append(result1[i])
        cluster2.append(result2[i])
        cluster3.append(result3[i])
        cluster4.append(result4[i])

    r = []
    for i in range(len(result)):
        count = 0
        if result[i] in cluster:
            count += 1
        if result1[i] in cluster1:
            count += 1
        if result2[i] in cluster2:
            count += 1
        if result3[i] in cluster3:
            count += 1
        if result4[i] in cluster4:
            count += 1
        if count >= 3:
            r.append(0)
        else:
            r.append(1)
            
    end = pd.DataFrame({'id': range(0,len(result)), 'label': r})
    end.to_csv(sys.argv[2], index = False)
