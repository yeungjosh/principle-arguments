import torch
import torch.nn as nn
import torch.nn.functional as F


# Previously Model1
class MatmulT(nn.Module):
    def __init__(self, word_dim, dropout=0.25):
        super(MatmulT, self).__init__()
        self.dr = nn.Dropout(dropout)
        self.att_proj = nn.Linear(word_dim, word_dim)

    def forward(self, sent_enc, parg_enc):
        sent_proj = self.att_proj(self.dr(sent_enc))  # (batch_size, 2*word_dim)
        attn = F.softmax(torch.mm(sent_proj, parg_enc.t()))  # (batch_size, num_parg)
        parg_out = torch.mm(attn, parg_enc)  # (batch_size, 2*word_dim)
        return parg_out


# Previously nonexistent
class LinearT(nn.Module):
    def __init__(self, word_dim, num_parg, dropout=0.25):
        super(LinearT, self).__init__()
        self.dr = nn.Dropout(dropout)
        self.att_proj = nn.Linear(word_dim, word_dim)
        self.fc = nn.Linear(num_parg, word_dim)
        self.act = nn.LeakyReLU()

    def forward(self, sent_enc, parg_enc):
        sent_proj = self.att_proj(self.dr(sent_enc))  # (batch_size, 2*word_dim)
        attn = F.softmax(torch.mm(sent_proj, parg_enc.t()))  # (batch_size, num_parg)
        out = self.fc(self.dr(attn))
        return self.act(out)


class BaselineT(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super(BaselineT, self).__init__()

    def forward(self, sents, pargs):
        return sents


class Model(nn.Module):
    def __init__(self, n_words, word_dim, output_dim, trans, targs=[], dropout=0.25, embed=True):
        super(Model, self).__init__()
        embed_dim = word_dim
        if embed:
            self.embedding = nn.Embedding(n_words, word_dim, padding_idx=0)
            self.encoder = nn.LSTM(word_dim, word_dim, batch_first=True, bidirectional=True)
            embed_dim = word_dim * 2
        self.embed = embed
        self.fc = nn.Linear(embed_dim, output_dim)
        self.T = trans(embed_dim, *targs, dropout=dropout)
        self.dr = nn.Dropout(dropout)

    def forward(self, sents, pargs):
        if self.embed:
            sents = self.embedding(sents)
            pargs = self.embedding(pargs)
            _, (sent_enc, _) = self.encoder(sents)
            _, (parg_enc, _) = self.encoder(pargs)
            sent_enc = torch.cat((sent_enc[0], sent_enc[1]), dim=-1)  # (batch_size, 2*word_dim)
            parg_enc = torch.cat((parg_enc[0], parg_enc[1]), dim=-1)  # (num_parg, 2*word_dim)
        else:
            sent_enc = sents
            parg_enc = pargs

        parg_enc = parg_enc / torch.norm(parg_enc, dim=1, keepdim=True)
        emb = self.T(sent_enc, parg_enc)
        output = F.log_softmax(self.fc(self.dr(emb)))
        return output
