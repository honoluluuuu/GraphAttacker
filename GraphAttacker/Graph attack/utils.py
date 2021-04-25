import numpy as np
import time
import scipy.sparse as sp
import tensorflow as tf
import random
import copy
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

#trasforma matrici in tuple
def to_tuple(mat):
    if not sp.isspmatrix_coo(mat):
        mat = mat.tocoo()
    idxs = np.vstack((mat.row, mat.col)).transpose()
    values = mat.data
    shape = mat.shape
    return idxs, values, shape

#trasforma matrici sparse in tuble
def sparse_to_tuple(sparse_mat):
    if isinstance(sparse_mat, list):
        for i in range(len(sparse_mat)):
            sparse_mat[i] = to_tuple(sparse_mat[i])
    else:
        sparse_mat = to_tuple(sparse_mat)
    return sparse_mat

#normalizza la matrice delle feature per riga e la trasforma in tupla
def process_features(features):
    features /= features.sum(1).reshape(-1, 1)
    features[np.isnan(features) | np.isinf(features)] = 0 #serve per le features dei nodi globali, che sono di soli 0.
    return sparse_to_tuple(sp.csr_matrix(features))

#renormalization trick della matrice di adiacenza
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return sp.csr_matrix(a_norm)


#conversione a tupla e normalizzazione della matrice d'adiacenza
def preprocess_adj(adj, is_gcn, symmetric = True):
    if is_gcn:
        adj = adj + sp.eye(adj.shape[0]) # ogni nodo ha come vicino anche se stesso, fa parte di GCN
    adj = normalize_adj(adj, symmetric)
    return sparse_to_tuple(adj)


#  --------------------- metriche --------------------------------------------
#cross-entropy con mascheramento per isolare i nodi con label
def masked_cross_entropy(predictions, labels, mask):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask) #per normalizzare la loss finale
    loss *= mask
    return tf.reduce_mean(loss)

#accuracy con mascheramento
def masked_accuracy(predictions, labels, mask):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

#  ----------------------- init -----------------------------------------------
#inizializzatore di pesi secondo Glorot&Bengio
def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    val = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(val, name=name)

def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

#costruzione del dizionario per GCN 
def build_dictionary_GCN(adj_batch, feats, support, pool_mat1, labels, labels_mask, att_adj,att_fea, att_labels,att_labels_mask, att_pool_mat, num_size, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['labels_mask']: labels_mask})
    dictionary.update({placeholders['feats'][i]: feats[i] for i in range(len(feats))})
    dictionary.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    dictionary.update({placeholders['num_features_nonzero'][i]: feats[i][1].shape for i in range(len(feats))})
    dictionary.update({placeholders['pool_mat1'][i]: pool_mat1[i] for i in range(len(pool_mat1))})
    dictionary.update({placeholders['adj_batch'][i]: adj_batch[i] for i in range(len(adj_batch))})
    dictionary.update({placeholders['num_size'][i]: num_size[i] for i in range(len(num_size))})
    # dictionary.update({placeholders['att_fea']: att_fea})
    dictionary.update({placeholders['att_labels']: att_labels})
    dictionary.update({placeholders['att_labels_mask']: att_labels_mask})
    dictionary.update({placeholders['att_adj']: att_adj})
    dictionary.update({placeholders['att_fea']: att_fea})
    dictionary.update({placeholders['att_num_features_nonzero']: att_fea[1].shape})
    dictionary.update({placeholders['att_pool_mat']: att_pool_mat})
    return dictionary

def build_dictionary_Att(adj,att_fea,  support, pool_mat, labels, labels_mask, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['labels_mask']: labels_mask})
    # dictionary.update({placeholders['batch']: batch})
    # dictionary.update({placeholders['feats']: feats})
    # dictionary.update({placeholders['feats']: feats})
    dictionary.update({placeholders['support']: support })
    dictionary.update({placeholders['num_features_nonzero']: att_fea[1].shape })
    dictionary.update({placeholders['pool_mat']: pool_mat})
    dictionary.update({placeholders['att_adj']: adj})
    dictionary.update({placeholders['att_fea']: att_fea})
    return dictionary

def build_dictionary_Att_X(adj,att_fea, att_fea_dense, support, pool_mat, labels, labels_mask, placeholders):
    #prepara il dizionario che sarà poi passato alla sessione di TF
    dictionary = dict()
    dictionary.update({placeholders['labels']: labels})
    dictionary.update({placeholders['labels_mask']: labels_mask})
    dictionary.update({placeholders['support']: support })
    dictionary.update({placeholders['num_features_nonzero']: att_fea[1].shape })
    dictionary.update({placeholders['pool_mat']: pool_mat})
    dictionary.update({placeholders['att_adj']: adj})
    dictionary.update({placeholders['att_fea']: att_fea})
    dictionary.update({placeholders['att_fea_dense']: att_fea_dense})
    return dictionary

def eval_C(support, adj):
    # support=support[0]
    # adj=adj[0]
    N = len(support[0])
    modify_num_total=0
    for i in range(N):
        for j in range(N):
            if i!=j:
                if support[i][j]!=adj[i][j]:
                    modify_num_total +=1
    modify_num_total=modify_num_total/2
    for i in range(N):
        for j in range(N):
            if i==j:
                if support[i][j]!=adj[i][j]:
                    modify_num_total += 1
    return modify_num_total

def eval_X(feature_ori, feature_attack):
    support=feature_ori
    x_attack=feature_attack
    N = support.shape[0]
    M = support.shape[1]
    modify_rata = 0
    modify_rata_zero = 0
    ori_num = 0
    zero_num = 0
    for i in range(N):
        for j in range(M):
            if support[i][j]!=0 and support[i][j]>1:
                # modify_rata +=abs((x_attack[i][j]-support[i][j])/support[i][j])
                modify_rata += abs((x_attack[i][j] - support[i][j]) ** 2 / support[i][j])
                ori_num +=1
            else:
                # if abs(x_attack[i][j])>1:
                    # modify_rata_zero +=abs(x_attack[i][j]/1)
                modify_rata += x_attack[i][j] ** 2 / 1
                zero_num +=1
    modify_rata = modify_rata/ori_num + modify_rata_zero/zero_num

    return modify_rata

def eval_X_onehot(feature_ori, feature_attack):
    support=feature_ori
    x_attack=feature_attack
    N = support.shape[0]
    M = support.shape[1]
    modify= 0
    for i in range(N):
        if np.argmax(support[i])!=np.argmax(x_attack[i]):
            modify +=1


    return modify


def eval_X_l2(feature_ori, feature_attack):
    support= copy.deepcopy(feature_ori)
    x_attack=copy.deepcopy(feature_attack)
    nor_ori = np.max(support, axis=0) - np.min(support,axis=0 )
    N = support.shape[0]
    M = x_attack.shape[1]
    nor_l2 = 0
    for i in range(N):
        for j in range(M):
            if nor_ori[j] != 0:
                support[i][j] /= nor_ori[j]
                x_attack[i][j] /= nor_ori[j]
            else:
                support[i][j] /= 1
                x_attack[i][j] /= 1
            nor_l2 += np.sqrt((support[i][j] - x_attack[i][j]) ** 2)
    nor_l2 /= M * N

    return nor_l2

def single_graph(num_graphs, num_nodes, idx, feats_matrix, ori_A):
    graph = []
    fea = []
    label = []
    ori_adj = []
    adj_size = np.zeros(num_graphs, dtype=int)
    for i in range(num_graphs - 1):
        adj_size[i] = idx[i + 1] - idx[i]
    adj_size[-1] = num_nodes - idx[num_graphs - 1]
    graph_size = max(adj_size)

    for k in range(num_graphs):
        graph.append(np.zeros((graph_size, graph_size)))
        fea.append(np.zeros((graph_size, feats_matrix.shape[1])))
        ori_adj.append(np.zeros((adj_size[k], adj_size[k])))
        if k < num_graphs - 1:
            for i in range(adj_size[k]):
                for j in range(adj_size[k]):
                    graph[k][i][j] = ori_A[i + idx[k]][j + idx[k]]
                    ori_adj[k][i][j] = ori_A[i + idx[k]][j + idx[k]]
                for l in range(feats_matrix.shape[1]):
                    fea[k][i][l] = feats_matrix[i + idx[k]][l]
    return graph, fea, label, ori_adj, adj_size, graph_size

def att_graph(attack_graph, graph, fea, labels, graph_size, adj_size):
    # att_pooling = np.array([[0. for i in range(graph_size)] for k in range(1)])
    att_pooling = np.zeros((graph_size, 130))
    ori_label = np.zeros((1, labels.shape[1]))
    att_label = np.zeros((1, labels.shape[1]))
    ori_class = np.nonzero(labels[attack_graph])[0][0]
    ori_label[0][ori_class] = 1
    num = 0
    while (num < 1):
        att = random.randint(0, att_label.shape[1] - 1)
        if att == ori_class:
            continue
        else:
            att_label[0][att] = 1
            num += 1
    att_mask = np.ones(1, dtype=bool)
    att_adj = graph[attack_graph]
    att_size = adj_size[attack_graph]
    att_fea_dense = fea[attack_graph]
    att_fea = process_features(csr_matrix(fea[attack_graph]))

    att_support = preprocess_adj(coo_matrix(att_adj), True, True)
    # att_pooling[0, range(0, att_size)] = (1 / att_size)

    for j in range(att_size):
        for k in range(130):
            att_pooling[j][k] = 1

    graph_link = 0
    graph_fea = 0
    for i in range(att_adj.shape[0]):
        for j in range(att_adj.shape[0]):
            if att_adj[i][j] == 1:
                graph_link += 1
        for k in range(att_fea_dense.shape[1]):
            if att_fea_dense[i][k]==1:
                graph_fea +=1
    graph_link /= 2
    return att_adj, att_fea, att_support, att_pooling, ori_class, ori_label, att_label, att_mask, graph_link, att_size, att_fea_dense, graph_fea

def data_rand(randnum, graph, fea, adj_size, labels):

    label1 = copy.deepcopy(labels)
    fea1 = copy.deepcopy(fea)
    graph1 = copy.deepcopy(graph)
    adj_size1 = copy.deepcopy(adj_size)
    label1 = label1.tolist()
    adj_size1 = adj_size1.tolist()

    random.seed(randnum)
    random.shuffle(graph1)
    random.seed(randnum)
    random.shuffle(fea1)
    random.seed(randnum)
    random.shuffle(label1)
    random.seed(randnum)
    random.shuffle(adj_size1)
    label1 = np.array(label1)
    adj_size1 = np.array(adj_size1)
    return graph1, fea1, label1, adj_size1


def data_batch(num_graphs, num_supports, graph_size, graph1, fea1, label1, adj_size1):
    label = []
    support = []
    features = []
    pool_mat = []
    pool_mat1 = []
    adj_batch = []
    num_nodes = []
    adj = copy.deepcopy(graph1)
    batch = 0

    for k in range(num_graphs):
        graph1[k] = coo_matrix(graph1[k])
        fea1[k] = csr_matrix(fea1[k])
    for k in range(num_graphs):
        if (k + 1) / num_supports > batch:
            label.append(np.zeros((num_supports, label1.shape[1])))
            support.append([])
            features.append([])
            pool_mat.append([])
            pool_mat1.append([])
            adj_batch.append([])
            num_nodes.append([])
            for i in range(num_supports):
                support[batch].append(preprocess_adj(graph1[batch * num_supports + i], True, True))
                features[batch].append(process_features(fea1[batch * num_supports + i]))
                adj_batch[batch].append(adj[batch * num_supports + i])
                pool_mat[batch].append(np.array([[0. for i in range(graph_size)] for k in range(1)]))
                pool_mat[batch][i][0, range(0, adj_size1[batch * num_supports + i])] = (
                            1 / adj_size1[batch * num_supports + i])

                pool_mat1[batch].append(np.zeros((graph_size, 130)))
                for j in range(adj_size1[batch * num_supports + i]):
                    for k in range(130):
                        pool_mat1[batch][i][j][k]=1

                num_nodes[batch].append(adj_size1[batch * num_supports + i])
                for j in range(label1.shape[1]):
                    label[batch][i][j] = label1[batch * num_supports + i][j]
            batch += 1
    return support, features, label, pool_mat, batch, adj_batch, num_nodes, pool_mat1

def get_txt(A,gen_dir):
    fw = open(gen_dir, 'w')
    for i in range(len(A[0])):
        if np.sum(A[i])!=0:
            for j in range(len(A[1])):
                if A[i][j] == 1:
                    fw.write(str(i) + " " + str(j) + '\n')
        # else:
        #     fw.write(str(i) + " " + str(i) + '\n')
    fw.close()


def att_graph1(attack_graph, graph, fea, labels, graph_size, adj_size):
    # att_pooling = np.array([[0. for i in range(graph_size)] for k in range(1)])
    att_pooling = np.zeros((graph_size, 130))
    ori_label = np.zeros((1, labels.shape[1]))
    att_label = np.zeros((1, labels.shape[1]))
    ori_class = np.nonzero(labels[attack_graph])[0][0]
    ori_label[0][ori_class] = 1
    num = 0
    while (num < 1):
        att = random.randint(0, att_label.shape[1] - 1)
        if att == ori_class:
            continue
        else:
            att_label[0][att] = 1
            num += 1
    att_mask = np.ones(1, dtype=bool)
    att_adj = graph[attack_graph]
    att_size = adj_size[attack_graph]
    att_fea_dense = fea[attack_graph]
    att_fea = fea[attack_graph]
    att_support = att_adj
    # att_support = preprocess_adj(coo_matrix(att_adj), True, True)
    # att_pooling[0, range(0, att_size)] = (1 / att_size)

    for j in range(att_size):
        for k in range(130):
            att_pooling[j][k] = 1

    graph_link = 0
    graph_fea = 0
    for i in range(att_adj.shape[0]):
        for j in range(att_adj.shape[0]):
            if att_adj[i][j] == 1:
                graph_link += 1
        for k in range(att_fea_dense.shape[1]):
            if att_fea_dense[i][k]==1:
                graph_fea +=1
    # graph_link /= 2
    return att_adj, att_fea, att_support, att_pooling, ori_class, ori_label, att_label, att_mask, graph_link, att_size, att_fea_dense, graph_fea

def data_batch1(num_graphs, num_supports, graph_size, graph1, fea1, label1, adj_size1):
    label = []
    support = []
    features = []
    pool_mat = []
    pool_mat1 = []
    adj_batch = []
    num_nodes = []
    adj = copy.deepcopy(graph1)
    batch = 0

    # for k in range(num_graphs):
        # graph1[k] = coo_matrix(graph1[k])
        # fea1[k] = csr_matrix(fea1[k])
    for k in range(num_graphs):
        if (k + 1) / num_supports > batch:
            label.append(np.zeros((num_supports, label1.shape[1])))
            support.append([])
            features.append([])
            pool_mat.append([])
            pool_mat1.append([])
            adj_batch.append([])
            num_nodes.append([])
            for i in range(num_supports):
                support[batch].append(graph1[batch * num_supports + i])
                # support[batch].append(preprocess_adj(graph1[batch * num_supports + i], True, True))
                features[batch].append(fea1[batch * num_supports + i])
                adj_batch[batch].append(adj[batch * num_supports + i])
                pool_mat[batch].append(np.array([[0. for i in range(graph_size)] for k in range(1)]))
                pool_mat[batch][i][0, range(0, adj_size1[batch * num_supports + i])] = (
                            1 / adj_size1[batch * num_supports + i])

                pool_mat1[batch].append(np.zeros((graph_size, 130)))
                for j in range(adj_size1[batch * num_supports + i]):
                    for k in range(130):
                        pool_mat1[batch][i][j][k]=1

                num_nodes[batch].append(adj_size1[batch * num_supports + i])
                for j in range(label1.shape[1]):
                    label[batch][i][j] = label1[batch * num_supports + i][j]
            batch += 1
    return support, features, label, pool_mat, batch, adj_batch, num_nodes, pool_mat1