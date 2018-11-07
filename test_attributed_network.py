'''
Reference implementation of network representation for AAAI 2019 paper.
Paper: Evolutionarily Learning Multi-aspect Interactions and Influences from Network Structure and Node Content
For more details, refer to the paper:
'''

import numpy as np
from model.attentionlayer import Attention, AverageAttention
from keras.constraints import max_norm
import keras.backend as K
from attributed_data.read_attributed_data import read_attribute_graph
from keras.models import Sequential, Model
from keras.layers import GRU, Embedding, concatenate, Input, Dense, Lambda, Masking, Add, add,average
from attributed_data.load_attributed_network import *
from model.dynet import DynamicNet
import pandas as pd
from attributed_data.evaluation import test_linkpredict,classification
from sklearn.model_selection import train_test_split
import config


def make_dict():
    idx = list(node_mapping_dic.values())
    stat_dict = None
    if model.state_emb:
        stat_dict = dict(zip(node_mapping_dic.keys(), model.state_emb.get_weights()[0][idx]))
    emb_dict = dict(zip(node_mapping_dic.keys(), model.get_features(idx, feat_mat=txt_array, batch_size=predict_batch_sz)))
    return emb_dict, stat_dict



def evaluation(classificate_task):
    if classificate_task:
        label = np.loadtxt(directory + '/label.txt', dtype=np.int32)
        label_order = np.zeros([node_size], dtype=np.int32)
        for i in range(len(label)):
            id = label[i, 0]
            if id in node_set:
                l = label[i, 1]
                label_order[node_mapping_dic[id]] = l

        X_train, X_test, y_train, y_test = train_test_split(range(node_size), label_order, train_size=train_size)
        node_embed = model.get_features(np.arange(0, node_size), feat_mat=txt_array, batch_size=batch_sz)
        x_train = node_embed[X_train]
        x_test = node_embed[X_test]
        classification(x_train, y_train, x_test, y_test)
    else:
        test_linkpredict(test_path, model.embed_score_model, embedding_dict, state_dict,
                         directed=True, batch_size=predict_batch_sz)




classificate_task = True

directory = "attributed_data/DBLPv7"
directed = True
weighted = False
if classificate_task:
    path = directory+'/adjedges.txt'
    train_size = 0.8
else:
    path = directory+'/dblpv7_train_0.1.edgelist'
    test_path = directory + '/dblpv7_test_negative_0.1.txt'

nx_G = read_attribute_graph(path,directed=directed, weighted=weighted)

node_set = set(nx_G.nodes)
node_size = len(node_set)

print(len(nx_G.edges))
print(node_size)

node_text_set = {}
node_mapping_dic = {}
node_inv_map_dic = {}

txt_array = np.zeros([node_size, max_sent_len])
for i, id in enumerate(node_set):
    words = annotation_map[str(id)]
    node_text_array = np.zeros([1, max_sent_len],dtype=np.int32)
    if len(words) > 0:
        node_text_array[0,-len(words):] = [vocab_dict[w] for w in words]
        node_text_set[id] = node_text_array
    node_mapping_dic[id] = i
    node_inv_map_dic[i] = id
    txt_array[i]= node_text_array



sent_input = Input((max_sent_len,), name='sent_input')
word_embed = Embedding(vocab_size + 1, embed_len, mask_zero=True, name='word_embedding',
                       weights=[embeddings], trainable=False)(sent_input)
# sent_embed = GRU(embed_len, activation='tanh')(word_embed)
sent_embed = Attention(alpha=4, keepdims=True)(word_embed)
# txtmodel.add(l2_norm)

node_input = Input((1,), name='node_input')
node_embed = Embedding(node_size, embed_len, name='node_embed')(node_input)

attributed_embed = add([sent_embed, node_embed])

attributed_embed = Lambda(lambda x : K.l2_normalize(x, axis=-1))(attributed_embed)

attributed_model = Model(inputs=[node_input,sent_input],outputs=attributed_embed)

model = DynamicNet(attributed_model, node_size, state_dim=10, directed=directed,
                    run_dynamics=True)

edges = list(nx_G.edges())
nb_edges = len(edges)
from_node = np.zeros([nb_edges], dtype=np.int32)
to_node = np.zeros([nb_edges], dtype=np.int32)

print(nb_edges)
for i,e in enumerate(edges):
    from_node[i]=node_mapping_dic[edges[i][0]]
    to_node[i] = node_mapping_dic[edges[i][1]]




dropout_ratio = config.dropout_ratio
predict_batch_sz=config.predict_batch_sz
batch_sz = config.batch_sz
margin_aspect = config.margin_aspect
margin_overall = config.margin_overall

update_net_cfg_per_n_iter = config.update_net_cfg_per_n_iter
max_iter= config.max_iter

for it in range(max_iter):
    if it % update_net_cfg_per_n_iter == 0:

        net_corrupt = model.dropout_network(from_node, to_node, values=None, ratio=dropout_ratio)
        print('Reconfig network')

    print('Iteration %d/%d'%(it+1, max_iter))
    ix = np.random.permutation(len(from_node))
    neg_spl_nodes = np.random.randint(0, node_size, size=len(from_node), dtype=np.int32)
    model.fit((from_node[ix], to_node[ix], neg_spl_nodes), txt_array, edges=net_corrupt,
              margin_con=margin_aspect, margin_overall=margin_overall,
              batch_size=batch_sz, pred_batch_size=predict_batch_sz)

    # if (it + 1) % 10 == 0:
    #     evaluation()

post_train_it = 20
net_corrupt = model.dropout_network(from_node, to_node, values=None, ratio=0)
for it in range(post_train_it):
    print('Post Iteration %d/%d' % (it + 1, post_train_it))
    ix = np.random.permutation(len(from_node))
    neg_spl_nodes = np.random.randint(0, node_size, size=len(from_node), dtype=np.int32)
    model.fit((from_node[ix], to_node[ix], neg_spl_nodes), txt_array, edges=net_corrupt,
              margin_con=margin_aspect, margin_overall=margin_overall,
              batch_size=batch_sz, pred_batch_size=predict_batch_sz)



embedding_dict, state_dict = make_dict()
embed_save_path = directory + '/embed'
np.save(embed_save_path, np.hstack([np.expand_dims(list(embedding_dict.keys()), axis=1), list(embedding_dict.values())]))

evaluation(classificate_task)
