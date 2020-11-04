import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import pickle

import model as m
from utils import pad_sents, data_iterator

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

import nltk

nltk.download('punkt')

# df = pd.read_csv('/content/drive/My Drive/577-project/reason-dataset.csv')
df = pickle.load(open('data/reasons_complete_df.pkl', 'rb'))
df['lower_line'] = df['line'].apply(lambda r: r.lower()).apply(word_tokenize)

CoPAs = pd.read_csv('data/principle_argument_CoPA/PA_list.txt', header=None, sep='$', squeeze=True)
copas_t = CoPAs.apply(lambda row: row.lower()).apply(word_tokenize)

copa_sbert = pickle.load(open('data/principle_argument_CoPA/PA_embeds_SBERT.pkl', 'rb'))
copa_tfidf = pickle.load(open('data/principle_argument_CoPA/PA_embeds_tfidf_df.pkl', 'rb'))
df_dan = pickle.load(open('daniel_reasons.pkl', 'rb'))
copa_dan = pickle.load(open('daniel_copa.pkl', 'rb'))

vocab = set()
for r in df['lower_line']:
    vocab.update(r)
for r in copas_t:
    vocab.update(r)
len(vocab)
word2ix = dict(zip(vocab, range(1, len(vocab) + 1)))
word2ix['<PAD>'] = 0


def one_epoch(model, opt, train_text, train_stance, test_text, test_stance, tensor_copas):
    ls = []
    model.train()
    for claim, stance in data_iterator(train_text, train_stance):
        input_vec = torch.tensor(pad_sents(claim))
        stance_vec = torch.tensor(stance)
        probs = model(input_vec.to(device), tensor_copas.to(device))
        loss = F.nll_loss(probs, stance_vec.to(device))
        loss.backward()
        opt.step()
        ls.append(loss.item())

    model.eval()
    with torch.no_grad():
        input_vec = torch.tensor(pad_sents(test_text))
        stance_vec = torch.tensor(test_stance)
        probs = model(input_vec.to(device), tensor_copas.to(device))
        tl = F.nll_loss(probs, stance_vec.to(device)).item()

        nwrong = np.count_nonzero(np.argmax(probs.cpu().numpy(), 1) - test_stance)
        acc = 1 - (nwrong / len(test_stance))
    return acc, np.mean(ls), tl


from sklearn.model_selection import KFold

text1 = df['lower_line'].apply(lambda r: [word2ix[w] for w in r]).to_numpy()
text2 = np.vstack(df['claim_SBERT_embedding'].tolist())
text3 = np.vstack(df_dan['encodedLine'].tolist())
text4 = np.vstack(df['claim_tf_idf_embedding'].tolist())
copa1 = copas_t.apply(lambda r: [word2ix[w] for w in r]).to_numpy()
copa2 = np.vstack(copa_sbert['CoPA_SBERT_embedding'].tolist())
copa3 = np.vstack(copa_dan)
copa4 = np.vstack(copa_tfidf['CoPA_embedding'].tolist())
stances = (df['stance'] == 'Pro').astype(int).to_numpy()
stances3 = (df_dan['stance'] == 'Pro').astype(int).to_numpy()

names = [
    'Scratch + Baseline',
    'Scratch + Attn',
    'Scratch + Decoder',
    'SBERT + Baseline',
    'SBERT + Attn',
    'SBERT + Decoder',
    'Infersent + Baseline',
    'Infersent + Attn',
    'Infersent + Decoder',
    'tf-idf + Baseline',
    'tf-idf + Attn',
    'tf-idf + Decoder',
]
varargs = [
    [len(vocab) + 1, 100, 2, m.BaselineT, [], .3, True],
    [len(vocab) + 1, 100, 2, m.MatmulT, [], .3, True],
    [len(vocab) + 1, 100, 2, m.LinearT, [len(copa1)], .3, True],
    [0, 768, 2, m.BaselineT, [], .5, False],
    [0, 768, 2, m.MatmulT, [], .5, False],
    [0, 768, 2, m.LinearT, [len(copa1)], .5, False],
    [0, 4096, 2, m.BaselineT, [], .5, False],
    [0, 4096, 2, m.MatmulT, [], .5, False],
    [0, 4096, 2, m.LinearT, [len(copa1)], .5, False],
    [0, 300, 2, m.BaselineT, [], .5, False],
    [0, 300, 2, m.MatmulT, [], .5, False],
    [0, 300, 2, m.LinearT, [len(copa1)], .5, False],
]
lrs = [
    1e-4,
    1e-4,
    1e-4,
    5e-6,
    5e-6,
    5e-6,
    1e-5,
    1e-6,
    1e-6,
    5e-8,
    5e-8,
    5e-8,
]
datas = [
    [text1, stances, copa1],
    [text1, stances, copa1],
    [text1, stances, copa1],
    [text2, stances, copa2],
    [text2, stances, copa2],
    [text2, stances, copa2],
    [text3, stances3, copa3],
    [text3, stances3, copa3],
    [text3, stances3, copa3],
    [text4, stances, copa4],
    [text4, stances, copa4],
    [text4, stances, copa4],
]

kf = KFold(5)
# out = display(IPython.display.Pretty('Starting'), display_id=True)
results = {}
for name, va, lr, dat in zip(names, varargs, lrs, datas):
    text, st, copa = dat
    copa = torch.tensor(pad_sents(copa))
    res = []
    for i, j in kf.split(text):
        torch.cuda.empty_cache()
        print(f'Iteration {len(results)}')
        model = m.Model(*va)
        model.to(device)
        torch.cuda.empty_cache()
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=.01)
        accs = []
        l1 = []
        l2 = []
        while len(accs) == 0 or len(accs) - np.argmax(accs) < 800:
            acc, tr_l, te_l = one_epoch(model, opt, text[i], st[i], text[j], st[j], copa)
            accs.append(acc)
            l1.append(tr_l)
            l2.append(te_l)
            # out.update(IPython.display.Pretty(f'model {len(results)}, fold {len(res)}, epoch {len(accs)}.\n'\
            # f'Max acc {np.amax(accs)} ix {np.argmax(accs)}\n'\
            # f'Cur acc {acc}, ls {tr_l}  {te_l}'))
        res.append({'acc': accs, 'train': l1, 'test': l2})
        break
    results[name] = res
    zz = [np.amax(r['acc']) for r in res]
    print(name)
    print(f'Mean accuracy across 5 folds: {np.mean(zz)}')
    print(f'Max accuracy across 5 folds: {np.amax(zz)}')

pickle.dump(results, open('kfold_result.pkl', 'wb'))
