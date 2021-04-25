from utils import *


def get_neighborhood_k(graph, source, k):

    neighborhood_k = []
    len_dir = nx.single_source_shortest_path_length(graph, source, cutoff=k)

    for key in len_dir:
        if len_dir[key] == k:
            neighborhood_k.append(key)

    return neighborhood_k


def get_sub_graph(graph, source, k, labels, features):

    sub_graph_node = []
    sub_node = []
    sub_labels = []
    sub_features = []

    for i in range(k):
        sub_graph_node.extend(get_neighborhood_k(graph, source, i))
    for i in range(k+1):
        sub_node.extend(get_neighborhood_k(graph, source, i))
    for i in range(len(sub_node)):
        sub_labels.append(labels[sub_node[i]])
        sub_features.append(features.A[sub_node[i]])

    sub_graph_node = list(set(sub_graph_node))
    adj = nx.adjacency_matrix(graph)
    sub_graph = nx.Graph()

    for k in range(len(sub_graph_node)):
        for i in range(adj.A.shape[0]):
            if adj.A[sub_graph_node[k]][i] == 1:
                sub_graph.add_edge(sub_graph_node[k], i, wight=1)

    sub_labels = np.array(sub_labels)
    sub_features = np.array(sub_features)
    sub_features = sp.lil_matrix(sub_features)
    sub_adj = nx.adjacency_matrix(sub_graph)

    return sub_adj, sub_labels, sub_features


def add_sub_node(sub, sub_list):
    for t in range(len(sub)):
        for y in range(len(sub[t])):
            if sub[t][y] not in sub_list:
                sub_list.append(sub[t][y])
    return sub_list

def get_sub(adj, i,n,subgra):
    subgra.append([])
    for j in range(adj.shape[0]):
        if adj[i][j]==1:
            subgra[n].append(j)


def get_sub_A(adj, sub,sub_n):
    sub_A = np.zeros((sub_n, sub_n)).astype('float32')
    for i in sub:
        index_i = sub.index(i)
        # print("第i个节点：",index_i,"   ",i )
        for j in range(adj.shape[0]):
            if adj[i][j] == 1:
                if j in sub:
                    index_j = sub.index(j)
                    sub_A[index_i][index_j] = 1
    return sub_A

def subgraph(adj,feature,labels, attack_node, node_class,istarget=False):
    feature = feature.A
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    add_random = 0
    sub_feature = []
    sub_label = []
    sub_num = []
    sub = []
    sub1 = []
    subgra_1 = []
    subgra_2 = []
    subgra_3 = []
    subgra_4 = []
    subgra_5 = []

    ori_class  = np.nonzero(labels[0])[0][0]
    for i in attack_node:
        get_sub(adj, i, n1, subgra_1)
        # while (add_random < 2):
        #     idx = random.randint(0, adj.shape[0] - 1)
        #     if idx not in sub:
        #         if istarget:
        #             if np.sum(labels[idx]) != 0:
        #                 if np.nonzero(labels[idx])[0][0] == node_class:
        #                     subgra_1[n1].append(idx)
        #                     add_random += 1
        #         else:
        #             if np.nonzero(labels[idx])[0][0] != ori_class:
        #                 subgra_1[n1].append(idx)
        #                 add_random += 1
        # 找二阶邻居
        for q in subgra_1[n1]:
            get_sub(adj, q, n2, subgra_2)
        #
        #     # 找三阶邻居
            for e in subgra_2[n2]:
                get_sub(adj, e, n3, subgra_3)
                # for w in subgra_3[n3]:
                #     get_sub(adj, w, n4, subgra_4)
                #     # for z in subgra_4[n4]:
                #     #     get_sub(adj, z, n5, subgra_5)
                #     #     n5 += 1
                #     n4 +=1
                n3 += 1
            n2 += 1
        n1 += 1
        # 添加中心节点
        for t in range(len(attack_node)):
            if attack_node[t] not in sub:
                sub.append(attack_node[t])
        # 添加一阶邻居节点
        # add_sub_node(subgra_1, sub)
        # add_sub_node(subgra_1, sub1)
        # sub_num.append(len(sub))
        # # 添加二阶邻居节点
        # add_sub_node(subgra_2, sub)
        # add_sub_node(subgra_2, sub1)
        # sub_num.append(len(sub))
        # # # 添加三阶邻居节点
        # add_sub_node(subgra_3, sub)
        # add_sub_node(subgra_3, sub1)
        # sub_num.append(len(sub))

        add_sub_node(subgra_1, sub)
        add_sub_node(subgra_1, sub1)
        sub1_num = len(sub)
        # 添加二阶邻居节点
        add_sub_node(subgra_2, sub)
        add_sub_node(subgra_2, sub1)
        sub2_num = len(sub)
        # 添加三阶邻居节点
        add_sub_node(subgra_3, sub)
        add_sub_node(subgra_3, sub1)
        sub3_num = len(sub)

        #
        # add_sub_node(subgra_4, sub)
        # sub_num.append(len(sub))
        # #
        # add_sub_node(subgra_5, sub)
        # sub_num.append(len(sub))
        #四阶
        # add_sub_node(subgra_4, sub1)
        # add_sub_node(subgra_4, sub)
        # if len(sub1)<200:
        #     add_sub_node(subgra_4, sub)
        #     add_sub_node(subgra_5, sub1)
        #     if len(sub1)<200:
        #         add_sub_node(subgra_5, sub)

        # #随机添加节点,数量为子图大小的20%
        if len(sub)//5<10:
            random_num = 10
        else:
            random_num = len(sub)//5
        while(add_random<random_num):
            x = random.randint(0,adj.shape[0]-1)
            if int(np.sum(labels[x]))==0:
                continue
            if x not in sub:
                if istarget:
                    if np.nonzero(labels[x])[0][0] == node_class:
                        sub.append(x)
                        add_random +=1
                else:
                    if np.nonzero(labels[x])[0][0] != ori_class:
                        sub.append(x)
                        add_random += 1

        sub_num1 = len(sub)
        sub_A = get_sub_A(adj, sub,  sub_num1)
        for i in sub:
            sub_feature.append(feature[i].tolist())
            sub_label.append(labels[i])
    sub_feature = np.array(sub_feature)
    sub_feature = sp.lil_matrix(sub_feature)
    sub_label = np.array(sub_label)
    sub_A = sp.csc_matrix(sub_A)

    # return subgra_1,subgra_2,subgra_3,sub_num,sub ,sub_A ,sub_feature,sub_label
    return sub, sub_A, sub_feature, sub_label ,sub1_num, sub2_num, sub3_num


def subgraph1(adj,feature,labels, attack_node, similar_node):
    feature = feature.A
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    add_random = 0
    sub_feature = []
    sub_label = []
    sub_num = []
    sub3_num = 0
    sub = []
    sub1 = []
    subgra_1 = []
    subgra_2 = []
    subgra_3 = []
    subgra_4 = []
    subgra_5 = []
    sim_in = []
    sim_not = []
    ori_class  = np.nonzero(labels[0])[0][0]
    for i in attack_node:
        get_sub(adj, i, n1, subgra_1)
        # 找二阶邻居
        for q in subgra_1[n1]:
            get_sub(adj, q, n2, subgra_2)
            # 找三阶邻居
            for e in subgra_2[n2]:
                get_sub(adj, e, n3, subgra_3)
                # for w in subgra_3[n3]:
                #     get_sub(adj, w, n4, subgra_4)
                #     for z in subgra_4[n4]:
                #         get_sub(adj, z, n5, subgra_5)
                #         n5 += 1
                #     n4 +=1
                n3 += 1
            n2 += 1
        n1 += 1
        # 添加中心节点
        for t in range(len(attack_node)):
            if attack_node[t] not in sub:
                sub.append(attack_node[t])
        # 添加一阶邻居节点
        add_sub_node(subgra_1, sub)
        add_sub_node(subgra_1, sub1)
        sub1_num = len(sub)
        # 添加二阶邻居节点
        add_sub_node(subgra_2, sub)
        add_sub_node(subgra_2, sub1)
        sub2_num = len(sub)
        # 添加三阶邻居节点
        add_sub_node(subgra_3, sub)
        add_sub_node(subgra_3, sub1)
        sub3_num = len(sub)


        #四阶
        # add_sub_node(subgra_4, sub1)
        # # add_sub_node(subgra_4, sub)
        # if len(sub1)<200:
        #     add_sub_node(subgra_4, sub)
        #     add_sub_node(subgra_5, sub1)
        #     if len(sub1)<200:
        #         add_sub_node(subgra_5, sub)

        #随机添加节点,数量为子图大小的20%
        for add_index in range(len(similar_node)):
            if similar_node[add_index] not in sub:
                sub.append(similar_node[add_index])
                sim_not.append(similar_node[add_index])
            else:
                sim_in.append(similar_node[add_index])

        # if len(sub)//5<10:
        #     random_num = 10
        # else:
        #     random_num = len(sub)//5
        # while(add_random<random_num):
        #     x = random.randint(0,adj.shape[0]-1)
        #     if int(np.sum(labels[x]))==0:
        #         continue
        #     if x not in sub:
        #         if np.nonzero(labels[x])[0][0] == node_class:
        #             sub.append(x)
        #             add_random +=1


        sub_num = len(sub)
        sub_A = get_sub_A(adj, sub,  sub_num)
        for i in sub:
            sub_feature.append(feature[i].tolist())
            sub_label.append(labels[i])
    sub_feature = np.array(sub_feature)
    sub_feature = sp.lil_matrix(sub_feature)
    sub_label = np.array(sub_label)
    sub_A = sp.csc_matrix(sub_A)

    # return subgra_1,subgra_2,subgra_3,sub_num,sub ,sub_A ,sub_feature,sub_label
    return sub, sub_A, sub_feature, sub_label,sub1_num, sub2_num, sub3_num #,sim_in ,sim_not

def get_A_all(A_sub,A_ori,sub,size):
    n=0
    A = np.zeros((size,size))
    for i in range(A_ori.shape[0]):
        for j in range(A_ori.shape[1]):
            A[i][j] = A_ori[i][j]
    for i in range(A_sub.shape[0]):
        for j in range(A_sub.shape[1]):
            index_i = sub[i]
            index_j = sub[j]
            if A[index_i][index_j]!=A_sub[i][j]:
                A[index_i][index_j]=A_sub[i][j]
            # else:
            #     A[i][j] = A_ori[i][j]
            #     print(i,"   ",j)

    return A