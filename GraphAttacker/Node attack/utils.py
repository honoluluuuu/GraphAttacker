import numpy as np
import tensorflow as tf
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
from tensorflow.contrib import layers
from pathlib import Path
import sys
import random


def load_npz1(file_name):    # load datasets | return A, X, labels
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None
        labels = loader.get('labels')
    return adj_matrix, attr_matrix, labels




def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """MASK."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str,num_node,num_label):
    labels = np.zeros((num_node, num_label))
    with open('data/'+str(dataset_str)+'_label.txt')as f:  #label text
        for j in f:
            entry = [float(x) for x in j.split(" ")]
            labels[int(entry[0]), int(entry[1])] = 1

    adj=np.zeros((num_node,num_node))
    with open('data/'+str(dataset_str)+'.txt')as f:        #original network
        # reader = csv.reader(f)
        for j in f:
            entry = [float(x) for x in j.split(" ")]
            adj[int(entry[0]),int(entry[1])]=1
            adj[int(entry[1]), int(entry[0])] = 1
    features=np.eye(num_node,dtype=float)
    features=lil_matrix(features)
    adj =csr_matrix(adj)
    idx_total = []
    with open(str(dataset_str)+'.txt')as f: #train node list
        for j in f:
            entry = [int(x) for x in j.split(" ")]
            idx_total.append(entry[0])
    idx_train = []
    with open(str(dataset_str)+'_train_node.txt')as f: #train node list
        for j in f:
            entry = [int(x) for x  in j.split(" ")]
            idx_train.append(entry[0])
    idx_val = []
    with open(str(dataset_str)+'_va_node.txt')as f:    #validation node list
        for j in f:
            entry = [int(x) for x in j.split(" ")]
            idx_val.append(entry[0])
    idx_test = []
    with open(str(dataset_str)+'_test_node.txt')as f: #test node list
        for j in f:
            entry = [int(x) for x in j.split(" ")]
            idx_test.append(entry[0])

    total_mask = sample_mask(idx_total, labels.shape[0])
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    # attack_mask = sample_mask(idx_attack, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)


    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features,labels

def load_npz(file_name):

    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            # attr_matrix = None
            attr_matrix = np.eye(adj_matrix.shape[0], dtype=float)
            attr_matrix = lil_matrix(attr_matrix)
        labels = loader.get('labels')
        labels1 = np.zeros((adj_matrix.shape[0], 2))
        for i in range(len(labels)):
            labels1[i][labels[i]] = 1

        label1, label2 = np.split(labels1, [758, ], 0)
        len1 = len(label1)
        len2 = len(label2)
        tra1 = int(len1 * 0.1)
        val1 = int(len1 * 0.2)
        tra2 = int(len2 * 0.1)
        val2 = int(len2 * 0.2)
        train_mask1 = sample_mask(range(tra1), label1.shape[0])
        val_mask1 = sample_mask(range(tra1, val1), label1.shape[0])
        test_mask1 = sample_mask(range(val1, len1), label1.shape[0])

        train_mask2 = sample_mask(range(tra2), label2.shape[0])
        val_mask2 = sample_mask(range(tra2, val2), label2.shape[0])
        test_mask2 = sample_mask(range(val2, len2), label2.shape[0])

        train_mask = np.concatenate([train_mask1, train_mask2])
        val_mask = np.concatenate([val_mask1, val_mask2])
        test_mask = np.concatenate([test_mask1, test_mask2])

        y_train = np.zeros(labels1.shape)
        y_val = np.zeros(labels1.shape)
        y_test = np.zeros(labels1.shape)
        y_train[train_mask, :] = labels1[train_mask, :]
        y_val[val_mask, :] = labels1[val_mask, :]
        y_test[test_mask, :] = labels1[test_mask, :]


    return adj_matrix, attr_matrix, labels1 ,y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data1(dataset_str):


    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # adj, features, labels1 = load_npz1('data/citeseer.npz')
        # labels = np.zeros((len(labels1), np.max(labels1)+1))
        # for i in range(len(labels1)):
        #     labels[i][labels1[i]] = 1
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask

def read_graph(input,weighted=False,directed=False):
	'''
	Reads the input network in networkx.
	'''
	if weighted:
		G = nx.read_edgelist(input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not directed:
		G = G.to_undirected()

	return G


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# def preprocess_features(features):
#     """Row-normalize feature matrix and convert to tuple representation"""
#     rowsum = np.array(features.sum(1))
#     #rowsum = float(rowsum)
#     # r_inv = np.power(rowsum, -1).flatten()
#     # r_inv[np.isinf(r_inv)] = 0.
#     # r_mat_inv = sp.diags(r_inv)
#     # features = r_mat_inv.dot(features)
#
#     # features.toarry()
#     #return sparse_to_tuple(features)
#     return rowsum
#     #return features

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    #rowsum = float(rowsum)
    r_inv = np.power(rowsum, -1,dtype=float).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    # features.toarry()
    return sparse_to_tuple(features)
    #return rowsum
    #return features

def preprocess_features1(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features_normalized = features
    # features.toarry()
    #return sparse_to_tuple(features)
    return features_normalized

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj_spase(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized =adj
    # adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))#A+IN
    return adj_normalized
    # return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask,attack, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['attack_node']: attack})

    return feed_dict

def construct_feed_dict_gcn(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})

    return feed_dict






def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def get_X(X):
    features = np.zeros(shape=(None,2))
    for i in X.shape[0]:
        for j in X.shape[1]:
            if X[i][j]==1.0:
                features[0]=i
                features[1]=j


def kl_gaussian_loss(mean, stddev, epsilon=1e-8):
    return (-0.5 * tf.reduce_mean(1.0 + tf.log(tf.square(stddev) + epsilon) - tf.square(stddev) -
                                    tf.square(mean)))

    eval_c = eval_C(support, support_Gen)
    if T == 1:
        cut = eval_c
        T=0
    if eval_c<cut*0.8:
        cut = eval_c
        #learning_rate = learning_rate*0.9

    print("      Gen_total_acc=",acc_gen_total,"   ",acc_g_t, "Ori_acc=", acc_ori,"modify_num",eval_c,"     ",cut,"     ",learning_rate)

def get_txt(A,gen_dir):
    fw = open(gen_dir, 'w')
    for i in range(len(A[0])):
        if np.sum(A[i])!=0:
            for j in range(len(A[1])):
                if A[i][j] == 1:
                    fw.write(str(i) + " " + str(j) + '\n')
        else:
            fw.write(str(i) + " " + str(i) + '\n')
    fw.close()

def cal_link_num(adj):
    All_link_num = 0
    for i in range(adj.shape[0]):  # 完整网络的连边数
        for j in range(adj.shape[1]):
            if adj[i][j] == 1:
                All_link_num += 1
    return All_link_num

def get_att_nodes(attack_str, labels, test_mask,y_test):
    Node = []
    att_txt = Path(attack_str)
    if att_txt.exists():
        with open(attack_str, 'r') as f:
            lines = f.readlines()
            for node in lines:
                node = int(node.split(' ')[0])
                Node.append(node)
    else:
        for i in range(test_mask.shape[0]):
            if test_mask[i]:
                star = i
                break

        Node = []
        with open(attack_str, 'w') as f:
            for c in range(labels.shape[1]):
                num = 50
                node_num = 0
                while node_num < num:
                    node = random.randrange(star, test_mask.shape[0])
                    if y_test[node][c]:
                        Node.append([node, c])
                        node_num += 1
                        f.write(str(node) + ' \n')
    return Node