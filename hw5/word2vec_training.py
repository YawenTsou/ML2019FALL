from gensim.models import word2vec
import pandas as pd
import spacy
import re
import pickle
import sys

def write(file, f, tokenizer):
    for i in range(len(file)):
        sent = file.loc[i, 'comment'].replace('@user ', '')
        sent = re.sub("[+\!\/\\_$%^*()+.,:\-\"“”]+|[+——！，。？、~@#￥%……&*（）：`]+", ' ', sent)
        sent = sent.replace('  ', ' ')
        sent = sent.lower()
        for j in tokenizer(sent):
            f.write(str(j))
            f.write(' ')
        f.write('\n')
    
if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_lg')
    tokenizer = spacy.lang.en.English().Defaults().create_tokenizer(nlp)
    
    train_x = pd.read_csv(sys.argv[1])
    test_x = pd.read_csv(sys.argv[2])
    
    f = open('corpus.csv', 'w')
    write(train_x, f, tokenizer)
    write(test_x, f, tokenizer)
    f.close()
    
    sentences = word2vec.LineSentence("corpus.csv")
    embedding = word2vec.Word2Vec(sentences=sentences, size=500, window=5, min_count=20)

    w2v = []
    for _, key in enumerate(embedding.wv.vocab):
        w2v.append((key, embedding.wv[key]))
    special_tokens = ["<PAD>", "<UNK>"]
    for token in special_tokens:
        w2v.append((token, [0.0] * 500))
        
    with open('w2v.pkl', 'wb') as f:
        pickle.dump(w2v, f)