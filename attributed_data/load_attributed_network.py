import networkx as nx
import math
from sklearn.utils import shuffle
import numpy as np
from collections import namedtuple
import json
from keras.preprocessing.text import text_to_word_sequence
from attributed_data.read_attributed_data import load_glove_embedding,create_json_data

dir= "attributed_data/DBLPv7"
# dir= "attributed_data/coauthor"
embed_len = 300
embedding_dir = "attributed_data/glove/"
embedding_path = embedding_dir + "glove.6B.%dd.txt" % embed_len
# embedding_path is the glove embedding path
# embed_len is set according to the download glove embedding
# Download glove embedding as the initial embedding in https://nlp.stanford.edu/projects/glove/

#create_json_data(dir) # you can generate text json file through this function and we already put it in the data file

with open(dir + "/vocab.json", 'r') as jsonfile:
    vocab_dict = json.load(jsonfile)

with open(dir + "/caption.json", 'r') as jsonfile:
    caption_dataset = json.load(jsonfile)
vocab_size = len(vocab_dict)
max_sent_len = caption_dataset['max_sent_len']
annotation_map = caption_dataset['annotation']
embeddings = load_glove_embedding(embedding_path, vocab_dict, embed_len)

