# This script creates sentence embeddings for reasons and copa's using the SBERT method.
# It saves the reasons embeddings into reasons_df.pkl and the copa embeddings into PA_embeds_SBERT.pkl

# SETUP: Install the model with pip:

# pip install -U sentence-transformers

# Repository by Nils Reimers Gregor Geigle fine-tunes BERT / RoBERTa / DistilBERT / ALBERT / XLNet
# https://github.com/UKPLab/sentence-transformers

import pandas as pd
import numpy as np
import pickle

import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize 
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer 

pickle_copadict = open("./principle_argument_CoPA/PA_dict.pkl","rb")
principle_args = pickle.load(pickle_copadict)
# print(principle_args
copa_df = pd.read_csv("./principle_argument_CoPA/IBM_Debater_(R)_CoPA-Motion-ACL-2019.v0.csv")
try:
    reasons_df = pd.read_pickle('reasons_df.pkl')
except (OSError, IOError) as e:
    reasons_df = pd.read_csv("./reason-dataset.csv")
reasons_df.head()

complete_df = pd.read_pickle('reasons_complete_df.pkl')
complete_df.head()

complete_df['avg_w2v_embedding'] = reasons_df['avg_w2v_embedding'].values
complete_df.head()
reasons_df=complete_df
# complete_df.to_pickle('reasons_complete_df.pkl') # contains BERT now
docs = reasons_df['line'].tolist()
docs[:2]
model = SentenceTransformer('bert-base-nli-mean-tokens')

reasons_embeddings = model.encode(docs)
# Sentence-BERT (SBERT) embeddings
reasons_df['claim_SBERT_embedding'] = reasons_embeddings
reasons_df.to_pickle('reasons_complete_df.pkl') # contains BERT now
CoPAs = pd.read_pickle(r'principle_argument_CoPA/PA_list.pkl')
model = SentenceTransformer('bert-base-nli-mean-tokens')
pa_embeddings = model.encode(CoPAs)

copa_embeds_tfidf_df = pd.DataFrame({'CoPAs': CoPAs, 'CoPA_SBERT_embedding': pa_embeddings})
copa_embeds_tfidf_df.to_pickle('./principle_argument_CoPA/PA_embeds_SBERT.pkl')
pd.read_pickle('./principle_argument_CoPA/PA_embeds_SBERT.pkl')


















