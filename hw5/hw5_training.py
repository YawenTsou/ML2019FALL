import pandas as pd
import numpy as np
import re
import spacy
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
    
class DialogDataset(Dataset):
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
        comment_lens = [len(data['comment']) for data in datas]
        idx = sorted(range(len(comment_lens)), key=lambda k: comment_lens[k], reverse=True)
        batch['id'] = [datas[x]['id'] for x in idx]
        batch['labels'] = torch.tensor([datas[x]['label'] for x in idx])

        batch['comment_lens'] = [comment_lens[x] if comment_lens[x] != 0 else 1 for x in idx]
        padded_len = min(self.comment_lens, max(batch['comment_lens']))
        tmp = []
        for i in idx:
            tmp.append(self.vocab.trim_pad(self.vocab.convert_tokens_to_indices(datas[i]['comment']), padded_len))
        batch['comment'] = torch.tensor(tmp)

        return batch

class LSTMNet(nn.Module):
    def __init__(self, pretrained_embedding):
        super(LSTMNet, self).__init__()
        
        pretrained_embedding = torch.FloatTensor(pretrained_embedding)
        self.embedding = nn.Embedding(
            pretrained_embedding.size(0),
            pretrained_embedding.size(1), padding_idx=1572)
        self.embedding.weight = torch.nn.Parameter(pretrained_embedding)
        
        self.lstm = nn.LSTM(500, 800, 3, dropout=0.2, bidirectional=True, batch_first=True)
        self.hidden2out = nn.Linear(800 * (1+True), 1)
        
    def forward(self, comment, lens):
        x = self.embedding(comment)
        comment_pack = rnn_utils.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        out, (h1, c1) = self.lstm(comment_pack)
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
        if out_len[0]-1 <= 0:
            out_layer = out_pad[0].mean(0).unsqueeze(0)
        else:
            out_layer = out_pad[0][:out_len[0]-1].mean(0).unsqueeze(0)
        for i in range(1, len(out_pad)):
            if out_len[i]-1 <= 0:
                out_layer = torch.cat((out_layer, out_pad[i].mean(0).unsqueeze(0)))
            else:
                out_layer = torch.cat((out_layer, out_pad[i][:out_len[i]-1].mean(0).unsqueeze(0)))
            
        y = self.hidden2out(out_layer)
               
        return y.flatten()


if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_lg')
    tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)
    
    train_x = pd.read_csv(sys.argv[1])
    train_y = pd.read_csv(sys.argv[2])
    
    with open('w2v_1213.pkl', 'rb') as f:
        w2v = pickle.load(f)
        
    trains = []
    for i in range(len(train_x)):
        train = {}
        train['id'] = train_x.loc[i, 'id']
        train['label'] = train_y.loc[i, 'label']
        tmp = []
        sent = train_x.loc[i, 'comment'].replace('@user ', '')
        sent = re.sub("[+\!\/\\_$%^*()+.,:\-\"“”]+|[+——！，。？、~@#￥%……&*（）：`]+", ' ', sent)
        sent = sent.replace('  ', ' ')
        sent = sent.lower()
        for j in tokenizer(sent):
            tmp.append(str(j))
        train['comment'] = tmp
        trains.append(train)    
        
    train_set = DialogDataset(trains, w2v)
    train_loader = Data.DataLoader(train_set, collate_fn=train_set.collate_fn, batch_size=32, shuffle = True)
    valid_set = DialogDataset(trains, w2v)
    valid_loader = Data.DataLoader(valid_set, collate_fn=train_set.collate_fn, batch_size=32, shuffle = False)    
    
    EPOCH = 15
    model = LSTMNet([x[1] for x in w2v])
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    acc_train_his = []
    loss_train_his = []
    for epoch in range(EPOCH):
        model.train()
        train_loss = []
        train_acc = []
        for idx, data in enumerate(train_loader):             
            if use_gpu:
                comment = data['comment'].cuda()
                lens = torch.tensor(data['comment_lens']).cuda()
                labels = data['labels'].float().cuda()

            optimizer.zero_grad()
            output = model(comment, lens)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            predict = torch.tensor([1 if x >=0.5 else 0 for x in output])
            acc = np.mean((data['labels'] == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, train Loss: {:.4f}, train accuracy: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        acc_train_his.append(np.mean(train_acc))
        loss_train_his.append(np.mean(train_loss))

        model.eval()
        with torch.no_grad():
            valid_acc = []
            for idx, data in enumerate(valid_loader):
                if use_gpu:
                    comment = data['comment'].cuda()
                    lens = torch.tensor(data['comment_lens']).cuda()
                    labels = data['labels'].float().cuda()

                output = model(comment, lens)
                predict = torch.tensor([1 if x >=0.5 else 0 for x in output])
                acc = np.mean((data['labels'] == predict).cpu().numpy())
                valid_acc.append(acc)
            print("Epoch: {}, valid accuracy: {:.4f}".format(epoch + 1, np.mean(valid_acc)))

        if np.mean(valid_acc) > 0.8:
            checkpoint_path = 'model_{}.pth'.format(epoch+1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)
    