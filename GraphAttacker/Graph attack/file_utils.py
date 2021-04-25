import numpy as np
import os
from collections import defaultdict
from itertools import groupby
import scipy.sparse as sp
import random
from scipy.sparse import csr_matrix
import copy
import sys

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return onehot


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_outer(dataset, model):
    if model == 'gcn':
        return load_data_gcn(dataset)


def add_one_by_one(l):
    new_l = []
    cumsum = 0
    for elt in l:
        cumsum += elt
        new_l.append(cumsum)
    return new_l



def load_data_gcn(dataset="cora"):
    print('Loading: {} dataset...'.format(dataset))

    feats_and_labels = np.genfromtxt("cora\{}.content".format(dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(feats_and_labels[:, 1:-1], dtype=np.int32)
    labels = encode_onehot(feats_and_labels[:, -1])

    # costruzione grafo
    idx = np.array(feats_and_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("cora\{}.cites".format(dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.int32)

    # simmetrizzazione della matrice d'adiacenza
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    print('{} has {} nodes, {} edges, {} features.'.format(dataset, adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, labels

def get_splits(labels, train_dim, val_dim, test_dim):
    train_ind = range(train_dim[1])
    val_ind = range(val_dim[0], val_dim[1])
    test_ind = range(test_dim[0], test_dim[1])

    labels_train = np.zeros(labels.shape, dtype=np.int32)
    labels_val = np.zeros(labels.shape, dtype=np.int32)
    labels_test = np.zeros(labels.shape, dtype=np.int32)

    labels_train[train_ind] = labels[train_ind]
    labels_val[val_ind] = labels[val_ind]
    labels_test[test_ind] = labels[test_ind]

    train_mask = sample_mask(train_ind, labels.shape[0])
    val_mask = sample_mask(val_ind, labels.shape[0])
    test_mask = sample_mask(test_ind, labels.shape[0])

    return labels_train, labels_val, labels_test, train_ind, val_ind, test_ind, train_mask, val_mask, test_mask

def get_splits_graphs_basic(num_graphs, labels, train_dim, val_dim, test_dim, oldidx):

    idx = np.array([i for i in range(num_graphs)])
    #idx = np.concatenate( (np.array([range(0,98)]), np.array([range(100,198)]), np.array([range(200,298)]), np.array([range(300,398)]), np.array([range(400,498)]), np.array([range(500,598)])), axis=None ) 
    #idx = np.concatenate( (idx, np.array(range(98,100)), np.array(range(198,200)), np.array(range(298,300)), np.array(range(398,400)), np.array(range(498,500)), np.array(range(598,600))), axis=None )
    random.shuffle(idx)
    
    train_ind = [idx[train_dim[0] : train_dim[1]]]
    val_ind = [idx[val_dim[0] : val_dim[1]]]
    test_ind = [idx[test_dim[0] : test_dim[1]]]

    labels_train = np.zeros(labels.shape, dtype=np.int32)
    labels_val = np.zeros(labels.shape, dtype=np.int32)
    labels_test = np.zeros(labels.shape, dtype=np.int32)

    labels_train[train_ind] = labels[train_ind]
    labels_val[val_ind] = labels[val_ind]
    labels_test[test_ind] = labels[test_ind]

    train_mask = sample_mask(train_ind, labels.shape[0])
    val_mask = sample_mask(val_ind, labels.shape[0])
    test_mask = sample_mask(test_ind, labels.shape[0])

    return labels_train, labels_val, labels_test, train_ind, val_ind, test_ind, train_mask, val_mask, test_mask


def get_splits_graphs(num_graphs, labels, train_dim, val_dim, test_dim, idx):

    idx_incr = np.array([i for i in range(num_graphs)])
    idx_incr = idx + idx_incr
    random.shuffle(idx_incr) #fondamentale

    train_ind = [idx_incr[train_dim[0] : train_dim[1]]]
    val_ind = [idx_incr[val_dim[0] : val_dim[1]]]
    test_ind = [idx_incr[test_dim[0] : test_dim[1]]]

    labels_train = np.zeros(labels.shape, dtype=np.int32)
    labels_val = np.zeros(labels.shape, dtype=np.int32)
    labels_test = np.zeros(labels.shape, dtype=np.int32)

    labels_train[train_ind] = labels[train_ind]
    labels_val[val_ind] = labels[val_ind]
    labels_test[test_ind] = labels[test_ind]

    train_mask = sample_mask(train_ind, labels.shape[0])
    val_mask = sample_mask(val_ind, labels.shape[0])
    test_mask = sample_mask(test_ind, labels.shape[0])

    return labels_train, labels_val, labels_test, train_ind, val_ind, test_ind, train_mask, val_mask, test_mask

def load_data_basic(num_nodes, num_graphs, num_classes, dim_feats, dataset_name):
    global_nodes_idx = find_insert_position(dataset_name)
    adj_matrix = build_adj_diag_basic(num_nodes, num_graphs, dataset_name)
    node_feats = build_feats_basic(num_nodes, num_graphs, dim_feats, dataset_name)
    graph_labels = build_labels_basic(num_graphs, num_classes, num_nodes, dataset_name)
    return adj_matrix, node_feats, graph_labels, global_nodes_idx

def load_data_basic_label(num_nodes, num_graphs, num_classes, dim_feats, dataset_name):
    global_nodes_idx = find_insert_position(dataset_name)
    adj_matrix = build_adj_diag_basic(num_nodes, num_graphs, dataset_name)
    # node_feats = build_feats_basic(num_nodes, num_graphs, dim_feats, dataset_name)
    graph_labels = build_labels_basic(num_graphs, num_classes, num_nodes, dataset_name)
    return adj_matrix,  graph_labels, global_nodes_idx

def load_data(num_nodes, num_graphs, num_classes, dim_feats, dataset_name):

    global_nodes_idx = find_insert_position(dataset_name)

    adj_matrix = build_adj_diag(num_nodes, num_graphs, global_nodes_idx, dataset_name)

    node_feats = build_feats_vertConc(global_nodes_idx, num_nodes, num_graphs, dim_feats, dataset_name)

    graph_labels = build_labels_vertConc(num_graphs, num_classes, num_nodes, global_nodes_idx, dataset_name) #codifica one-hot
    
    return adj_matrix, node_feats, graph_labels, global_nodes_idx


def find_insert_position(dataset_name):
    path = dataset_name +"/"+dataset_name.upper()+"_graph_indicator.txt"
    node_ind = np.genfromtxt(path)
    fake_idx = np.where(node_ind[:-1] != node_ind[1:])[0]
    fake_idx = fake_idx + 1
    fake_idx = np.insert(fake_idx, 0, 0)
    return fake_idx

def build_labels_basic(graphs, num_classes, num_nodes, dataset_name ):
    if(os.path.exists(dataset_name +"_labels_basic.npy")):
            labels = np.load(dataset_name +"_labels_basic.npy")
            return labels

    path=dataset_name +"/"+dataset_name.upper()+"_graph_labels.txt"
    true_labels = np.loadtxt(path, dtype='i', delimiter=',') #prova
    labels = encode_onehot(true_labels)
    np.save(dataset_name +"_labels_basic.npy", labels)
    return labels

def build_labels_vertConc(graphs, num_classes, num_nodes, idx, dataset_name ):
    
    if(os.path.exists(dataset_name +"_labels_aug.npy")):
        labels = np.load(dataset_name +"_labels_aug.npy")
        return labels

    path=dataset_name +"/"+dataset_name.upper()+"_graph_labels.txt"

    true_labels = np.loadtxt(path, dtype='i', delimiter=',') #prova
    true_labels = encode_onehot(true_labels)
    labels = np.array([[0 for i in range(num_classes)] for k in range(num_nodes)])

    #inserisco le label vere su cui calcolare loss e verificare i risultati
    for i in range(graphs):
        labels = np.insert(labels, idx[i]+i, true_labels[i], axis=0)
        print("row: ")
        print(i)

    np.save(dataset_name +"_labels_aug.npy", labels)
    return labels

def build_feats_basic(num_nodes, graphs, dim_feats, dataset_name):
    if(os.path.exists(dataset_name+"_feats_matrix_basic.npz")):
        feats_matrix = sp.load_npz(dataset_name+"_feats_matrix_basic.npz")
        return feats_matrix
    else:
        if(os.path.exists(dataset_name +"/"+dataset_name.upper()+"_node_attributes.txt" )):
            path=dataset_name +"/"+dataset_name.upper()+"_node_attributes.txt"

            feats_matrix = np.loadtxt(path, delimiter=',')
            feats_matrix = sp.csr_matrix(feats_matrix)
            sp.save_npz(dataset_name+"_feats_matrix_basic", feats_matrix)
        else:
            path = dataset_name +"/"+dataset_name.upper()+"_node_labels.txt"
            node_l = []
            f_l = open(path, 'r')
            for line in f_l.readlines():
                curLine = line.strip("\n")
                val = int(curLine)
                node_l.append(val)
            fea_size = max(node_l) + 1
            fea_label = np.zeros((num_nodes, fea_size))
            for i in range(len(node_l)):
                fea_label[i][node_l[i]] = 1
            feature = csr_matrix(fea_label)
            feats_matrix = fea_label
    return feature

def build_feats_vertConc(idx, num_nodes, graphs, dim_feats, dataset_name):
    if(os.path.exists(dataset_name+"_feats_matrix_aug.npz")):
        feats_matrix = sp.load_npz(dataset_name+"_feats_matrix_aug.npz")
        return feats_matrix

    path=dataset_name +"/"+dataset_name.upper()+"_node_attributes.txt" 

    feats_matrix = np.loadtxt(path, delimiter=',')
    fake_feats = np.array([[0. for i in range(dim_feats)] for k in range(graphs)])
    #inserimento righe aggiuntive
    print("inserting global node rows:")
    for i in range(graphs):
        feats_matrix = np.insert(feats_matrix, idx[i]+i, fake_feats[i], axis=0)
        print("row: ")
        print(i)

    feats_matrix = np.absolute(feats_matrix)
    feats_matrix = sp.csr_matrix(feats_matrix)
    sp.save_npz(dataset_name+"_feats_matrix_aug", feats_matrix)
    return feats_matrix

def build_feats_vertConc_mean_features(idx,num_nodes, graphs, dim_feats, dataset_name):
    if(os.path.exists(dataset_name+"_feats_matrix_aug_with_mean.npz")):
        feats_matrix = sp.load_npz(dataset_name+"_feats_matrix_aug_with_mean.npz")
        return feats_matrix

    path=dataset_name +"/"+dataset_name.upper()+"_node_attributes.txt" 
    aug_idx = np.append(idx, int(num_nodes-1))

    feats_matrix = np.loadtxt(path, delimiter=',')
    #inizializzo le features di ogni nodo globale come la media delle features dei nodi del relativo grafo
    fake_feats = np.array([[ np.mean(feats_matrix[[ range(aug_idx[k],aug_idx[k+1]) ] ,[i] ] ) for i in range(dim_feats)] for k in range(graphs)])
    #inserimento righe aggiuntive
    print("inserting global node rows:")
    for i in range(graphs):
        feats_matrix = np.insert(feats_matrix, idx[i]+i, fake_feats[i], axis=0)
        print("row: ")
        print(i)

    feats_matrix = sp.csr_matrix(feats_matrix)
    sp.save_npz(dataset_name+"_feats_matrix_aug_with_mean", feats_matrix)
    return feats_matrix

def build_adj_diag_basic(nodes, graphs, dataset_name):
    if(os.path.exists(dataset_name +"_adj_matrix_basic.npz")):
        adj_matrix = sp.load_npz(dataset_name+"_adj_matrix_basic.npz")
        return adj_matrix
    
    path=dataset_name +"/"+dataset_name.upper()+"_A.txt"
    tmpdata = np.genfromtxt(path, dtype=np.dtype(str))
    ind1 = tmpdata[:, 1]
    ind2 = tmpdata[:, 0]
    adj_matrix = [[0 for i in range(nodes)] for k in range(nodes)]
    # adj_matrix = np.zeros((nodes, nodes))
    for i in range(len(ind1)):
        print(i)
        u = ind1[i]
        v = ind2[i]
        u = int(u)      #vanno letti come stringhe
        v = int(v[:-1]) #aggiustamenti per eliminare la virgola del file 
        adj_matrix[u-1][v-1] = 1 

    adj_matrix = np.matrix(adj_matrix)
    adj_matrix = sp.coo_matrix(adj_matrix)
    sp.save_npz(dataset_name + "_adj_matrix_basic", adj_matrix)
    return adj_matrix

#dal file ENZYMES_A costruisce una matrice diagonale a blocchi contenente le matrici di adiacenza di ogni grafo (un grafo per blocco)
def build_adj_diag(nodes, graphs, idx, dataset_name):
    
    if(os.path.exists(dataset_name +"_adj_matrix_aug.npz")):
        adj_matrix = sp.load_npz(dataset_name+"_adj_matrix_aug.npz")
        return adj_matrix
    
    path=dataset_name +"/"+dataset_name.upper()+"_A.txt"

    #vanno preparati 600 nodi globali, uno per grafo
    nodes_tot = nodes+graphs
    fake_matrix = np.array([[0 for i in range(nodes)] for k in range(graphs)])
    
    #preparazione righe aggiuntive per i nodi globali
    node_ind = np.genfromtxt(dataset_name+"/"+dataset_name.upper()+"_graph_indicator.txt")
    node_ind = node_ind.tolist()
    occ = [len(list(group)) for key, group in groupby(node_ind)]
    occ = add_one_by_one(occ) # serve per riempire la parte aggiunta della matrice di adiacenza
    occ.insert(0, 1)
    ranges = list(zip(occ[1:], occ))
    upper_idx, lower_idx = map(list, zip(*ranges))

    for index in range(graphs):
        fake_matrix[index][(lower_idx[index] -1) : (upper_idx[index]-1)] = 1

    #lettura matrice d'adiacenza originale
    print("parsing original adj matrix")
    tmpdata = np.genfromtxt(path, dtype=np.dtype(str))
    ind1 = tmpdata[:, 1]
    ind2 = tmpdata[:, 0]
    adj_matrix = [[0 for i in range(nodes)] for k in range(nodes)]
    for i in range(len(ind1)):
        u = ind1[i]
        v = ind2[i]
        u = int(u)      #vanno letti come stringhe
        v = int(v[:-1]) #aggiustamenti per eliminare la virgola del file 
        adj_matrix[u-1][v-1] = 1 
    
    #inserimento righe aggiuntive
    print("inserting global node rows:")
    for i in range(graphs):
        adj_matrix = np.insert(adj_matrix, idx[i]+i, fake_matrix[i], axis=0)
        print("row: ")
        print(i)

    #preparazione colonne aggiuntive per i nodi globali
    lower_idx_new = [0 for i in range(graphs)]
    upper_idx_new = [0 for i in range(graphs)]  
    for i in range(graphs):
        lower_idx_new[i] = lower_idx[i]+i #riscalo l'indice perchè ho inserito 600 nuove righe
        upper_idx_new[i] = upper_idx[i]+i #riscalo l'indice perchè ho inserito 600 nuove righe

    vert_padding = np.array([[0 for i in range(nodes_tot)] for k in range(graphs)])
    for i in range(graphs):
        vert_padding[i][(lower_idx_new[i]-1):(upper_idx_new[i]-1)] = 1

    #inserimento colonne aggiuntive
    print("inserting global node columns:")
    for i in range(graphs):
        adj_matrix = np.insert(adj_matrix, idx[i]+i, vert_padding[i], axis=1)
        print("column: ")
        print(i)


    adj_matrix = np.matrix(adj_matrix)
    adj_matrix = sp.coo_matrix(adj_matrix)
    sp.save_npz(dataset_name + "_adj_matrix_aug", adj_matrix)
    return adj_matrix


def get_data(num_nodes, num_graphs, num_classes, num_feats, dataset_name):
    idx = find_insert_position(dataset_name)
    labels_ori = build_labels_basic(num_graphs, num_classes, num_nodes, dataset_name)
    labels = copy.deepcopy(labels_ori)
    adj_size = np.zeros(num_graphs, dtype=int)
    for i in range(num_graphs - 1):
        adj_size[i] = idx[i + 1] - idx[i]
    adj_size[-1] = num_nodes - idx[num_graphs - 1]
    a_size = copy.deepcopy(adj_size)
    for i in range(len(adj_size)):
        if a_size[i] > 800:
            a_size[i] = 0
    graph_size = max(a_size)
    a = 0
    tobig = []
    path = dataset_name + "/" + dataset_name.upper() + "_A.txt"
    tmpdata = np.genfromtxt(path, dtype=np.dtype(str))
    ind1 = tmpdata[:, 1]
    ind2 = tmpdata[:, 0]  # [:-1]
    # ind3 = tmpdata[:, 0]

    path_indicator = dataset_name + "/" + dataset_name.upper() + "_graph_indicator.txt"
    f_indi = open(path_indicator, 'r')
    indicator = []
    for line in f_indi.readlines():
        curLine = line.strip().split(" ")
        indicator.append(int(curLine[0]) - 1)

    for i in range(len(adj_size)):
        if adj_size[i] > 800:
            tobig.append(i)
            a += 1
    adj = []
    feature = []
    # class_num = labels.sum(axis=0)
    f_l = open("DD/DD_node_labels.txt", 'r')
    node_l = []
    for line in f_l.readlines():
        curLine = line.strip("\n")
        val = int(curLine)
        node_l.append(val)
    fea_size = max(node_l) + 1

    for i in range(num_graphs):
        # if i not in tobig:
        adj.append(np.zeros((graph_size, graph_size)))
        feature.append(np.zeros((graph_size, fea_size)))
    g_start = 0

    for i in range(len(node_l)):
        cur_fea = indicator[i]
        if cur_fea not in tobig:
            feature[cur_fea][i - idx[cur_fea]][node_l[i]] = 1

    for i in range(len(ind1)):
        u = ind1[i]
        v = ind2[i]
        u = int(u) - 1  # vanno letti come stringhe
        v = int(v[:-1]) - 1
        cur_graph = indicator[u]
        if cur_graph not in tobig:
            adj[cur_graph][u - idx[cur_graph]][v - idx[cur_graph]] = 1
    delte = 0
    for i in range(len(tobig)):
        del adj[tobig[i] - delte]
        del feature[tobig[i] - delte]
        labels = np.delete(labels, tobig[i] - delte, axis=0)
        a_size = np.delete(a_size, tobig[i] - delte, axis=0)
        delte += 1
        # feature = np.delete(feature, tobig[i], axis=0)

    feats_matrix = feature
    feat = csr_matrix(feature[0])

    graph = copy.deepcopy(adj)

    return graph, feats_matrix, labels, idx, a_size, feat, graph_size