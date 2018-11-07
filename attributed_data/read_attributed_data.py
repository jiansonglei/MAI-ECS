import networkx as nx
import math
from sklearn.utils import shuffle
import numpy as np
from collections import namedtuple
import json
from keras.preprocessing.text import text_to_word_sequence

def read_attribute_graph(path, weighted=False, directed=False, remove_isolate=True):
    '''
    Reads the input unweighted network in networkx.
    directed or undirected
    '''
    if not directed:
        if weighted:
            G = nx.read_weighted_edgelist(path, nodetype=int)
        else:
            G = nx.read_adjlist(path, nodetype=int)
    else:
        if weighted:
            G = nx.read_weighted_edgelist(path, nodetype=int, create_using=nx.DiGraph())
        else:
            G = nx.read_adjlist(path, nodetype=int, create_using=nx.DiGraph())

    # if weighted:
    #     #G = nx.read_edgelist(path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    #     G = nx.read_weighted_edgelist(path, nodetype=int, create_using=nx.DiGraph())
    # else:
    #     G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
    #     # for edge in G.edges():
    #     #     G[edge[0]][edge[1]]['weight'] = 1
    #
    # if not directed:
    #     G = G.to_undirected()

    if remove_isolate:
        G.remove_nodes_from(list(nx.isolates(G)))

    return G

def clear_wordseq(wordseq, max_len=100):
    # Remove words less than 3 letters
    wordseq = [word for word in wordseq if len(word) > 1 ]
    # Remove numbers
    wordseq = [word for word in wordseq if not word.isdigit() and word.isalnum()]

    if len(wordseq)>max_len: wordseq = wordseq[0:max_len]

    return wordseq

def create_json_data(dir): #dir, directory of network dataset

    voca = set()
    ix_token = {}
    max_len = 35
    sen_len = []

    with open(dir+'/docs.txt') as f1:

        for i,l1 in enumerate(f1):
            tokens = text_to_word_sequence(l1)
            words = clear_wordseq(tokens[1:], max_len)
            id = tokens[0]
            voca |= set(words)
            ix_token[id] = words
            sen_len.append(len(words))

    vocabulary = {w: i for i, w in enumerate(voca, 1)}
    vocabulary_file = dir + "/vocab.json"
    with open(vocabulary_file, 'w') as jsonfile:
        json.dump(vocabulary, jsonfile)

    processed_file = dir + "/caption.json"
    with open(processed_file, 'w') as jsonfile:
        json.dump({'max_sent_len': max(sen_len),
                   'annotation': ix_token}, jsonfile)

def load_glove_embedding(embedding_path, dict, embeding_len):
    embeddings_index = {}
    f = open(embedding_path, 'r') #, encoding="utf8"
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding
    f.close()

    embedding_matrix = np.random.random([len(dict) + 1, embeding_len])
    embedding_matrix.astype(dtype=np.float32)
    for word, i in dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
#
# # dir= "dblp"
# dir = 'coauthor'
# dir= "DBLPv7/"
# # nx_G = read_attribute_graph(dir, weighted=False, directed=True)
# create_json_data(dir)