# This script creates sentence embeddings for reasons and copa's using the average Word2Vec method.
# It saves the reasons embeddings into reasons_df.pkl and the copa embeddings into PA_embeds_avg_w2v.pkl

import pandas as pd
import numpy as np
import pickle

import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from nltk.tokenize import sent_tokenize, word_tokenize 

pickle_copadict = open("./principle_argument_CoPA/PA_dict.pkl","rb")
principle_args = pickle.load(pickle_copadict)
# print(principle_args)
copa_df = pd.read_csv("./principle_argument_CoPA/IBM_Debater_(R)_CoPA-Motion-ACL-2019.v0.csv")
copa_df

wv_from_bin = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin",limit=50000, binary=True)

def get_avg_vector(w2v_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in w2v_model.vocab]
    if len(words) >= 1:
        return np.mean(w2v_model[words], axis=0)
    else:
        return []

def embed_sentence(s):
    if s[-1] == '.':
        s=s[:-1]
    data = [] 
    words = s.split()
    avg = get_avg_vector(wv_from_bin, words)
    return avg
# embed_sentence('It is a fact that there are differences between people. Hence, there should sometimes be differences in the way people are treated.')
try:
    reasons_df = pd.read_pickle('reasons_df.pkl')
except (OSError, IOError) as e:
    reasons_df = pd.read_csv("./reason-dataset.csv")
reasons_df.head()

CoPAs = pd.read_pickle(r'principle_argument_CoPA/PA_list.pkl')
CoPAs[:2]
CoPAs_embedding = [] # works
for arg in CoPAs:
    CoPAs_embedding.append(embed_sentence(arg.lower()))
copa_embeds_avg_w2v_df = pd.DataFrame({'CoPAs': CoPAs, 'CoPA_embedding': CoPAs_embedding})

copa_embeds_avg_w2v_df.head()
copa_embeds_avg_w2v_df.to_pickle('./principle_argument_CoPA/PA_embeds_avg_w2v.pkl')
def measure_pa_similarity(dataframe):
    CoPAs_embedding = []
    for arg in CoPAs:
        CoPAs_embedding.append(embed_sentence(arg))
    def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    similarity_scores = []
    for index, row in dataframe.iterrows():
        sentence=row['line']
        claim_embedding = embed_sentence(sentence)
        similarity = []
        for arg in CoPAs_embedding:
            similarity.append(cosine(claim_embedding, arg))
        similarity_scores.append(similarity)
    # similarity_scores is a list of a list of 74 embeddings for each line
    # len(similarity_scores[0]) ==  74
    dataframe['avg_w2v_embedding'] = pd.Series(similarity_scores)
    return dataframe
reasons_df=measure_pa_similarity(reasons_df)

reasons_df.to_pickle('reasons_df.pkl')
pd.read_pickle('reasons_df.pkl')








