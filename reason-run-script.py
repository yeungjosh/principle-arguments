import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

import model as m
from utils import pad_sents, data_iterator

df = pd.read_csv('reason-dataset.csv')
df['lower_line'] = df['line'].apply(lambda r: r.lower()).apply(word_tokenize)
CoPAs = pd.read_csv('principle_argument_CoPA/PA_list.txt', header=None, sep='$', squeeze=True)
copas_t = CoPAs.apply(lambda row: row.lower()).apply(word_tokenize)

vocab = set()
for r in df['lower_line']:
    vocab.update(r)
for r in copas_t:
    vocab.update(r)

word2ix = dict(zip(vocab, range(1, len(vocab)+1)))
word2ix['<PAD>']= 0

text = df['lower_line'].apply(lambda r: [word2ix[w] for w in r]).tolist()
copas = copas_t.apply(lambda r: [word2ix[w] for w in r]).tolist()
stances = (df['stance'] == 'Pro').astype(int).tolist()

model = m.Model1(len(vocab) + 1, 100, 2)
optimizer = optim.Adam(model.parameters(), weight_decay=.01)

tensor_copas = torch.tensor(pad_sents(copas))
n_iter = 200
for it in range(n_iter):
    # for claim, stance in data_iterator(corrected, stances):
    ls = []
    model.train()
    for claim, stance in data_iterator(text, stances):
        input_vec = torch.tensor(pad_sents(claim))
        stance_vec = torch.tensor(stance)
        probs = model(input_vec.to(device), tensor_copas.to(device))
        loss = F.nll_loss(probs, stance_vec.to(device))
        loss.backward()
        optimizer.step()
        ls.append(loss.item())

    print(f'epoch {it} done.')
    print(f'Training loss avg. {np.mean(ls)}')

    model.eval()
    with torch.no_grad():
        input_vec = torch.tensor(pad_sents(text))
        stance_vec = torch.tensor(stances)
        probs = model(input_vec.to(device), tensor_copas.to(device))
        l = F.nll_loss(probs, stance_vec.to(device)).item()
        print(f'Test loss: {l}')
        
        nwrong = np.count_nonzero(np.argmax(probs.cpu().numpy(), 1) - stances)
        print(f'Test %wrong: {nwrong / len(stances)}')

