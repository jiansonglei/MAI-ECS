import math
from keras.models import Model
from keras.layers import dot, Input

import numpy as np
from ranking import precisionAtK,recallAtK,nDCG,mrrAtK,avgPrecisionAtK
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
clf = LinearSVC()

#from data_utilities import split_rec_data_neg
def cn_scores(G,from_node,to_node):
    common_neighbor = set(G.neighbors(from_node)).intersection(set(G.neighbors(to_node)))
    score = len(common_neighbor)
    return score

def score_connection(from_rep,to_rep):
    return np.inner(from_rep,to_rep)

def score_model(score_model,memory,from_node,to_node):
    score = score_model.predict_aspect_scores([memory[from_node], memory[to_node]])
    return score

def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / min(float(k),len(act_set))
    return result

def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

def create_inner_score_model(dim):
    inputA = Input(dim, name='A')
    inputB = Input(dim, name='B')
    result = dot([inputA, inputB])
    return Model(inputs=[inputA, inputB], outputs=result)

def classification(x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, pos_label=None, average='macro')
    micro_f1 = f1_score(y_test, y_pred, pos_label=None, average='micro')
    print('Classification Accuracy=%f, macro_f1=%f, micro_f1=%f' % (acc, macro_f1, micro_f1))



def test_linkpredict(test_path, scoremodel, embedding_dict, state_dict=None, directed=False, batch_size=1000):
    test_data = np.loadtxt(test_path, dtype=np.int32)

    k = np.arange(1, 11)
    ap_vector = []
    recall_vector = []
    ndcg_vector = []

    target_nodes, test_nodes = test_data[:, 0], test_data[:, 1:]
    nb_case = len(test_nodes)
    test_case_len = test_nodes.shape[-1]
    batch_size = max(int(batch_size / test_case_len), 1)

    scores = {t: ([], []) for t in set(target_nodes)}
    all_scores = []

    multoutput = len(scoremodel.outputs) > 1

    nb_batch = int(math.ceil(float(nb_case) / batch_size))
    for i in range(nb_batch):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, nb_case)

        f_node = target_nodes[start_idx:end_idx]
        t_node = test_nodes[start_idx:end_idx]
        f_node_list = []
        t_node_list = []
        for f, t in zip(f_node, t_node):
            tl = list(t)
            t_node_list += tl
            f_node_list += [f] * len(tl)

        from_embed = [embedding_dict[j] for j in f_node_list]
        to_embed = [embedding_dict[j] for j in t_node_list]
        inputs = [from_embed, to_embed]

        if state_dict:
            from_state_embed = [state_dict[j] for j in f_node_list]
            to_state_embed = [state_dict[j] for j in t_node_list]
            inputs += [to_state_embed] if directed else [from_state_embed, to_state_embed]

        score_single = scoremodel.predict_on_batch(inputs)
        if multoutput:
            score_single = score_single[0]

        score_mat = np.reshape(score_single, [-1, test_case_len])
        all_scores.append(score_mat)
        for r in range(len(f_node)):
            poslist, neglist = scores[f_node[r]]
            poslist.append(score_mat[r, 0])
            neglist += list(score_mat[r, 1:])

        if i%1000 == 0:
            print('Prediction %d/%d' % (i + 1, nb_batch))

    for pscore, nscore in scores.values():
        ap_vector.append(avgPrecisionAtK(pscore, nscore, k))
        ndcg_vector.append(nDCG(pscore, nscore, k))
        recall_vector.append(recallAtK(pscore, nscore, k))

    mean_ap = np.mean(np.asarray(ap_vector), axis=0)
    mean_recall = np.mean(np.asarray(recall_vector), axis=0)
    mean_ndcg = np.mean(np.asarray(ndcg_vector), axis=0)

    all_scores = np.vstack(all_scores)
    recover_rate = []
    for row in all_scores:
        recover_rate.append(recallAtK([row[0]], row[1:], k))
    recover_rate = np.mean(np.asarray(recover_rate), axis=0)

    print("the mean average precision at")
    print(mean_ap)
    print("the mean recall at ")
    print(mean_recall)
    print("the mean ndcg at")
    print(mean_ndcg)

    print("the recovery rate")
    print(recover_rate)

