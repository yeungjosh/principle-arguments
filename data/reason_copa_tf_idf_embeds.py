# This script creates sentence embeddings for reasons and copa's using the weighted TF-IDF Word2Vec method.
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

from sklearn.feature_extraction.text import TfidfVectorizer 

pickle_copadict = open("./principle_argument_CoPA/PA_dict.pkl","rb")
principle_args = pickle.load(pickle_copadict)
# print(principle_args)
copa_df = pd.read_csv("./principle_argument_CoPA/IBM_Debater_(R)_CoPA-Motion-ACL-2019.v0.csv")
copa_df

wv_from_bin = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin",limit=50000, binary=True)

try:
    reasons_df = pd.read_pickle('reasons_df.pkl')
except (OSError, IOError) as e:
    reasons_df = pd.read_csv("./reason-dataset.csv")
reasons_df.head()

docs = reasons_df['line'].tolist()
docs[:2]
# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True, sublinear_tf=True)
# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)
(tfidf_vectorizer_vectors.shape[0])
tfidf_vocab = tfidf_vectorizer.vocabulary_
type(tfidf_vocab)

def get_avg_vector_tfidf(w2v_model, vocab, words):
    # remove out-of-vocabulary words
    words = [vocab[word] * w2v_model[word] for word in words if (word in w2v_model.vocab and word in vocab.keys())]
    if len(words) >= 1:
        return np.mean(words, axis=0)
    else:
        return []

def embed_sentence(s, vocab):
    if s[-1] == '.':
        s=s[:-1]
    words = s.split()
    return get_avg_vector_tfidf(wv_from_bin, vocab, words)
test = (embed_sentence('It is a fact that there are differences between people. Hence, there should sometimes be differences in the way people are treated.', tfidf_vocab))

CoPAs = pd.read_pickle(r'principle_argument_CoPA/PA_list.pkl')
CoPAs[:2]

# settings that you use for count vectorizer will go here
tfidf_vectorizer_copa=TfidfVectorizer(use_idf=True, sublinear_tf=True)
# just send in all your docs here
tfidf_vectorizer_copa_vectors=tfidf_vectorizer_copa.fit_transform(CoPAs)
tfidf_vectorizer_copa_vectors.shape
tfidf_copa_vocab = tfidf_vectorizer_copa.vocabulary_

CoPAs_embedding = [] # works
for arg in CoPAs:
    CoPAs_embedding.append(embed_sentence(arg.lower(), tfidf_copa_vocab))

copa_embeds_tfidf_df = pd.DataFrame({'CoPAs': CoPAs, 'CoPA_embedding': CoPAs_embedding})
copa_embeds_tfidf_df.to_pickle('./principle_argument_CoPA/PA_embeds_tfidf_df.pkl')

def cosine(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
def measure_pa_similarity(dataframe):
#     CoPAs_embedding = [] # works
#     for arg in CoPAs:
#         CoPAs_embedding.append(embed_sentence(arg.lower(), tfidf_copa_vocab))
# #     print(CoPAs_embedding)
    claim_embeddings = []
    similarity_scores = []
    for index, row in dataframe.iterrows():
        sentence=row['line']
        claim_embedding = embed_sentence(sentence.lower(), tfidf_vocab)
#         print('claim_embedding ',claim_embedding)
        claim_embeddings.append(claim_embedding)
        similarity = []
        for arg in CoPAs_embedding:
            similarity.append(cosine(claim_embedding, arg))
        similarity_scores.append(similarity)
    # similarity_scores is a list of a list of 74 embeddings for each line
    # len(similarity_scores[0]) ==  74
    dataframe['claim_tf_idf_embedding'] = pd.Series(claim_embeddings)
    dataframe['tf_idf_pa_similarity'] = pd.Series(similarity_scores)
    
    return dataframe, CoPAs_embedding
reasons_df, copa_embeddings = measure_pa_similarity(reasons_df)

reasons_df.to_pickle('reasons_df.pkl')
reasons_df = pd.read_pickle('reasons_df.pkl')











