import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset
import torch.utils.data as Data
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import pickle
import sys

class Vocab():
    def __init__(self, w2v):
        self._idx2token = [token for token, _ in w2v]
        self._token2idx = {token: idx for idx,
                           token in enumerate(self._idx2token)}
        self.PAD, self.UNK = self._token2idx["<PAD>"], self._token2idx["<UNK>"]

    def trim_pad(self, tokens, seq_len):
        return tokens[:min(seq_len, len(tokens))] + [self.PAD] * (seq_len - len(tokens))

    def convert_tokens_to_indices(self, tokens):
        return [
            self._token2idx[token]
            if token in self._token2idx else self.UNK
            for token in tokens]

    def __len__(self):
        return len(self._idx2token)
    
class TestDataset(Dataset):
    def __init__(self, data, w2v):
        self.data = data
        self.vocab = Vocab(w2v)
        self.comment_lens = 300

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(self.data[index])
        
    def collate_fn(self, datas):
        batch = {}
        
        # collate lists
        batch['id'] = [data['id'] for data in datas]
        batch['comment_lens'] = [len(data['comment']) for data in datas]
        padded_len = min(self.comment_lens, max(batch['comment_lens']))
        tmp = []
        for i in datas:
            tmp.append(self.vocab.trim_pad(self.vocab.convert_tokens_to_indices(i['comment']), padded_len))
        batch['comment'] = torch.tensor(tmp)

        return batch

class DNN(nn.Module):
    def __init__(self, pretrained_embedding):
        super(DNN, self).__init__()
        
        pretrained_embedding = torch.FloatTensor(pretrained_embedding)
        self.embedding = nn.Embedding(
            pretrained_embedding.size(0),
            pretrained_embedding.size(1), padding_idx=1515)
        self.embedding.weight = torch.nn.Parameter(pretrained_embedding)
        
        self.hidden2out = nn.Sequential(
            nn.Linear(500, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Linear(800, 1),
            nn.ReLU(),
        )

    def forward(self, comment, lens):
        x = self.embedding(comment[0][:lens[0]]).mean(0).unsqueeze(0)
        for i in range(1, len(comment)):
            if lens[i] <= 0:
                x = torch.cat((x, self.embedding(comment[i][-1].unsqueeze(0))))
            else:
                x = torch.cat((x, self.embedding(comment[i][:lens[i]]).mean(0).unsqueeze(0)))
            
        y = self.hidden2out(x)  
        return y.flatten()

if __name__ == '__main__':
    test_x = pd.read_csv(sys.argv[1])
    
    with open('w2v_final.pkl', 'rb') as f:
        w2v = pickle.load(f)
        
    tests = []
    for i in range(len(test_x)):
        test = {}
        test['id'] = test_x.loc[i, 'id']
        tmp = []
        sent = test_x.loc[i, 'comment'].replace('@user ', '')
        sent = re.sub("[+\!\/\\_$%^*()+.,:\-\"“”]+|[+——！，。？、~@#￥%……&*（）：`]+", ' ', sent)
        sent = sent.replace('  ', ' ')
        for j in tokenizer(sent):
            tmp.append(str(j))
        test['comment'] = tmp
        tests.append(test)
    
    model = DNN([x[1] for x in w2v])
    model.load_state_dict(torch.load('model_14_BOW.pth'))
    
    test_set = TestDataset(tests, w2v)
    test_loader = Data.DataLoader(test_set, collate_fn=test_set.collate_fn, batch_size=1, shuffle = False)
    
    model.eval()
    with torch.no_grad():
        end = []
        for idx, data in enumerate(test_loader):            
            output = model(data['comment'], torch.tensor(data['comment_lens']))
            predict = [1 if x >=0.5 else 0 for x in output]
            end += predict
            
    end = pd.DataFrame({'id': [x['id'] for x in tests], 'label': end})
    end.to_csv(sys.argv[2], index = False)