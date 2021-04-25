import tensorflow as tf
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from sklearn.manifold import TSNE
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    preds = tf.convert_to_tensor(preds)
    labels = tf.convert_to_tensor(labels)
    loss = labels * -tf.log(preds) +(1-labels) * -tf.log(1-preds)

    return tf.reduce_mean(loss)

def masked_sigmoid_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_accuracy1(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all), tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1)),tf.argmax(preds, 1),tf.argmax(labels, 1)


def eval_AML(support, adj):
    support=support[0]
    adj=adj[0]
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



def ave_sim(adj,sim, sub):
    index = []
    sim_list= []
    ave_sim = 0
    link_num = 0
    l_n = 0
    for v_t in range(len(adj)):
        if adj[0][v_t]==1:
            for i in range(len(sim)):
                if sim[i][0] == sub[v_t]:
                    index.append(i)
                    sim_list.append(sim[i][2])
                    if sim[i][2]>0 and sim[i][2]<=1:
                        ave_sim +=sim[i][2]
                        l_n +=1
            link_num +=1
    if l_n==0:
        AS = 0
    else:
        AS = ave_sim/l_n


    return index, sim_list, AS


def compute_alpha(n, S_d, d_min):
    return n / (S_d - n * np.log(d_min - 0.5)) + 1

def update_Sx(S_old, n_old, d_old, d_new, d_min):
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min

    d_old_in_range = np.multiply(d_old, old_in_range)
    d_new_in_range = np.multiply(d_new, new_in_range)

    new_S_d = S_old - np.log(np.maximum(d_old_in_range, 1)).sum(1) + np.log(np.maximum(d_new_in_range, 1)).sum(1)
    new_n = n_old - np.sum(old_in_range, 1) + np.sum(new_in_range, 1)
    return new_S_d, new_n

def compute_log_likelihood(n, alpha, S_d, d_min):
    return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d


def degree(adj,adj_gen):
    N = adj.shape[0]
    # Setup starting values of the likelihood ratio test.
    degree_sequence_start = adj.sum(0).A1
    current_degree_sequence = adj_gen.sum(0).A1
    d_min = 2
    S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
    S_d_gen = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))

    n_start = np.sum(degree_sequence_start >= d_min)
    n_gen = np.sum(current_degree_sequence >= d_min)

    alpha_start = compute_alpha(n_start, S_d_start, d_min)
    alphas_new = compute_alpha(n_gen, S_d_gen, d_min)
    alphas_combined = compute_alpha(n_gen + n_start, S_d_gen + S_d_start, d_min)

    log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)
    log_likelihood_gen = compute_log_likelihood(n_gen, alphas_new, S_d_gen, d_min)
    new_ll_combined = compute_log_likelihood(n_gen + n_start, alphas_combined, S_d_gen + S_d_start, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (log_likelihood_gen + log_likelihood_orig)
    return new_ratios
