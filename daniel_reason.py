import pandas as pd
import torch
from infersent_model import InferSent
import nltk
import torch.optim as optim
import numpy as np
import io
from gensim.models import Word2Vec, KeyedVectors
import torch.nn as nn
import pickle
from gensim.scripts.glove2word2vec import glove2word2vec


class NeuralNet(nn.Module):
    def __init__(self, dimension):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(dimension, 1)
        self.act1 = nn.Sigmoid()
        '''self.fc1 = nn.Linear(dimension, 20000)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(20000, 3000)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(3000, 2000)
        self.act3 = nn.ReLU()
        self.fc4 = nn.Linear(2000, 100)
        self.act4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(100, 1)
        self.act5 = nn.Sigmoid()'''

    def forward(self, input):
        a1 = self.fc1(input)
        h1 = self.act1(a1)
        '''a2 = self.fc2(h1)
        h2 = self.act2(a2)
        a3 = self.fc3(h2)
        h3 = self.act3(a3)
        a4 = self.fc4(h3)
        h4 = self.act4(a4)
        a5 = self.fc5(h4)
        h5 = self.act5(a5)'''
        return h1

def parse_reason():
    df = pd.read_csv("data/reason-dataset.csv", header=0, sep=',')
    return df[['stance', 'line']]

def measure_similarity():
    # Gathering sentences for encoding (the claims in the
    # 'train split' and all the principled arguments)
    dataframe = parse_reason()
    dataframe['line'] = dataframe['line'].apply(lambda r:r.lower())


    embeddings_dict = {}
    with open("GloVe/glove.840B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], "float32")
            except(ValueError):
                continue
            embeddings_dict[word] = vector

    all_sentences_vectorized = np.zeros([1, 300])
    labels = []
    for index, row in dataframe.iterrows():
        if row['stance'] == 'Pro':
            labels.append(1)
        else:
            labels.append(0)
        line = row['line']
        line = line.strip('"')
        line = line.strip("'")
        line = line.lower()
        split = line.split()

        counter = 0
        vectorized_sentence = np.zeros(300)
        for word in split:
            counter += 1
            try:
                vectorized_sentence = vectorized_sentence + embeddings_dict[word]
            except KeyError:
                vectorized_sentence = vectorized_sentence + np.zeros(300)
        counter = float(counter)
        vectorized_sentence /= counter
        all_sentences_vectorized = np.vstack([all_sentences_vectorized, vectorized_sentence])

    all_sentences_vectorized = np.delete(all_sentences_vectorized, 0, 0)

    copa_embeddings = np.zeros([1,300])
    CoPAs = pd.read_pickle(r'data/principle_argument_CoPA/PA_list.pkl')
    for line in CoPAs:
        line = line.strip('"')
        line = line.strip("'")
        line = line.lower()
        split = line.split()

        counter = 0
        vectorized_copa = np.zeros(300)
        for word in split:
            counter += 1
            try:
                vectorized_copa = vectorized_copa + embeddings_dict[word]
            except KeyError:
                vectorized_copa = vectorized_copa + np.zeros(300)
        counter = float(counter)
        vectorized_copa /= counter
        copa_embeddings = np.vstack([copa_embeddings, vectorized_copa])

    copa_embeddings = np.delete(copa_embeddings, 0, 0)


    return all_sentences_vectorized, copa_embeddings, labels


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def baseline_classifier():
    #all_sentences_vectorized, copa_embeddings, labels = measure_similarity()
    all_sentences_vectorized = pickle.load(open("dummy.p", "rb"))
    copa_embeddings = pickle.load(open("dummy1.p", "rb"))
    labels = pickle.load(open("dummy2.p", "rb"))

    train_input = all_sentences_vectorized[0:2000]
    test_input = all_sentences_vectorized[2000:2848]
    train_output = labels[0:2000]
    test_output = labels[2000:2848]


    net = NeuralNet(300)
    opt = optim.AdamW(net.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(50):

        opt.zero_grad()
        pred = net(torch.from_numpy(train_input).float())
        opt.zero_grad()
        loss = criterion(pred, torch.from_numpy(np.asarray(train_output)).float().unsqueeze(1))
        loss.backward()
        opt.step()

    correct = 0
    for row_index in range(0, len(test_output)):
        pred = net(torch.from_numpy(test_input[row_index]).float())

        if pred <= 0.5 and test_output[row_index] == 0:
            correct += 1
        elif pred >= 0.5 and test_output[row_index] == 1:
            correct += 1
    print(float(correct)/len(test_output))

def COPA_classifier():
    #all_sentences_vectorized, copa_embeddings, labels = measure_similarity()
    #pickle.dump(all_sentences_vectorized, open("dummy.p", "wb"))
    #pickle.dump(copa_embeddings, open("dummy1.p", "wb"))
    #pickle.dump(labels, open("dummy2.p", "wb"))

    all_sentences_vectorized = pickle.load(open("dummy.p", "rb"))
    copa_embeddings = pickle.load(open("dummy1.p", "rb"))
    labels = pickle.load(open("dummy2.p", "rb"))
    copa_distances = np.zeros([1,74])
    count = 0
    for row in all_sentences_vectorized:
        count +=1
        temp_copa = []
        for embeddings in copa_embeddings:
            temp_copa.append(cosine(row, embeddings))

        copa_distances = np.vstack([copa_distances, np.asarray(temp_copa)])

    copa_distances = np.delete(copa_distances, 0, 0)

    train_input = copa_distances[0:2000]
    test_input = copa_distances[2000:2848]
    train_output = labels[0:2000]
    test_output = labels[2000:2848]


    net = NeuralNet(74)
    opt = optim.AdamW(net.parameters(), lr=0.11)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(100):
        opt.zero_grad()
        pred = net(torch.from_numpy(np.asarray(train_input)).float())
        opt.zero_grad()
        loss = criterion(pred, torch.from_numpy(np.asarray(train_output)).float().unsqueeze(1))
        loss.backward()
        opt.step()

    correct = 0
    for row_index in range(0, len(test_output)):
        pred = net(torch.from_numpy(np.asarray(test_input[row_index])).float())

        if pred <= 0.5 and test_output[row_index] == 0:
            correct += 1
        elif pred >= 0.5 and test_output[row_index] == 1:
            correct += 1
    print(float(correct) / len(test_output))

if __name__ == '__main__':
    #measure_similarity()
    baseline_classifier()

    COPA_classifier()
