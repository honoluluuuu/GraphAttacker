from model import *

def get_similar_node(embedding, v_t, label , min, max):
    label_node = []
    similar_score = []
    target_label = 0
    for i in range(len(label[0])):
        label_node.append([])
        similar_score.append(0)
    similar_node = []
    sim_it = np.zeros(shape=(embedding.shape[0],3))
    for v_i in range(embedding.shape[0]):
        sim_it[v_i][0] = v_i
        sim_it[v_i][1] = np.nonzero(label[v_i])[0][0]
        sim_it[v_i][2] = cos_sim(embedding[v_t], embedding[v_i])
    data = sim_it[np.lexsort(-sim_it.T)]
    label_node[int(data[0][1])].append(int(data[0][0]))
    for i in range(data.shape[0]):
        if data[i][1]!=data[0][1] and data[i][2]>min and data[i][2]<max:
            label_node[int(data[i][1])].append(int(data[i][0]))
            similar_score[int(data[i][1])]+=data[i][2]
    max_len = len(label_node[0])
    for i in range(len(label_node)):
        if len(label_node[i])>max_len:
            max_len=len(label_node[i])
            target_label = i
        if similar_score[i]!=0:
            similar_score[i] = similar_score[i]/len(label_node[i])
    max_score = similar_score[0]
    for i in range(len(label_node[target_label])):
        if len(similar_node)<50:
            similar_node.append(label_node[target_label][i])

    return similar_node, target_label,data




def cos_sim(vector_a, vector_b):
    """
    :param vector_a:
    :param vector_b:
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def eucliDist(vector_a,vector_b):
    dist = np.sqrt(sum(np.power((vector_a - vector_b), 2)))
    eucli_sim = np.exp(-0.15*dist)
    return eucli_sim

