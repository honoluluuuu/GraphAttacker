import tensorflow as tf
import keras as k
from utils import *

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self, **kwargs):
        layer = self.__class__.__name__.lower()
        name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.weights = {}
        self.sparse_inputs = False

    def _call(self, inputs, support, num_features_nonzero, dim):
        return inputs

    def __call__(self, inputs, support, num_features_nonzero, dim):
        with tf.name_scope(self.name):
            outputs = self._call(inputs, support, num_features_nonzero, dim)
            return outputs


class ConvolutionalLayer(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout,
                 sparse_inputs, activation, isLast=False, bias=False, featureless=False, **kwargs):
        super(ConvolutionalLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.featureless = featureless
        self.activation = activation
        # self.support = [placeholders['support'][0]]
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_weights'):
            # for i in range(len(self.support)):
            for i in range(1):
                self.weights['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                           name='weights_' + str(i))
            if self.bias:
                self.weights['bias'] = zeros([output_dim], name='bias')


    def _call(self, inputs, support, num_features_nonzero, dim):
        x = inputs
        # dropout
        if self.sparse_inputs:
            # x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = sparse_dropout(x, 1 - self.dropout, num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        # for i in range(1):

        if not self.featureless:
            pre_sup = dot(x, self.weights['weights_' + str(0)],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.weights['weights_' + str(0)]

        self.support1 = self.preprocess_support(support, dim)
        support = dot(self.support1, pre_sup, sparse=False)
        supports.append(support)
        output = tf.add_n(supports)

        if self.bias:
            output += self.weights['bias']


        # y = tf.matmul(support,x)
        # y = tf.matmul(y, self.weights['weights_' + str(0)])
        # if self.bias:
        #     y += self.weights['bias']
        # output = tf.nn.l2_normalize(y,dim=1)

        return self.activation(output)

    def preprocess_support(self, adj, dim):
        In = tf.constant(1, shape=[dim], dtype=tf.float32)
        adj_normalized = self.nomalize_adj(tf.add(adj, tf.diag(In)),dim)
        return adj_normalized

    def nomalize_adj(self, adj, dim):
        D = tf.reduce_sum(adj, 1)
        D1 = tf.diag(tf.pow(D, tf.constant(-0.5, shape=[dim])))
        return tf.matmul(tf.matmul(D1, adj), D1)


class ConvolutionalLayer_1(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout,
                 sparse_inputs, activation, isLast=False, bias=False, featureless=False, **kwargs):
        super(ConvolutionalLayer_1, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.featureless = featureless
        self.activation = activation
        # self.support = [placeholders['support'][0]]
        self.sparse_inputs = sparse_inputs
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_weights'):
            # for i in range(len(self.support)):
            for i in range(1):
                self.weights['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                           name='weights_' + str(i))
            if self.bias:
                self.weights['bias'] = zeros([output_dim], name='bias')
            # self.weights['weights_' + str(i)] = glorot([input_dim, output_dim],
            #                                             name='weights_' + str(i))
            # if self.bias:
            #     self.weights['bias'] = zeros([output_dim], name='bias')

    def _call(self, inputs, support, num_features_nonzero, dim):
        x = inputs
        # dropout
        if self.sparse_inputs:
            # x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = sparse_dropout(x, 1 - self.dropout, num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        if not self.featureless:
            pre_sup = dot(x, self.weights['weights_' + str(0)],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.weights['weights_' + str(0)]
        self.support1 = self.preprocess_support(support)
        support = dot(self.support1, pre_sup, sparse=False)
        # support = dot(support, pre_sup, sparse=False)

        supports.append(support)
        output = tf.add_n(supports)

        if self.bias:
            output += self.weights['bias']

        #
        # y = tf.matmul(support,x)
        # y = tf.matmul(y, self.weights['weights_' + str(0)])
        # if self.bias:
        #     y += self.weights['bias']
        # output = tf.nn.l2_normalize(y,dim=1)

        return self.activation(output)

    def preprocess_support(self, adj):
        In = tf.constant(1, shape=[adj.shape[0]], dtype=tf.float32)
        adj_normalized = self.nomalize_adj(tf.add(adj, tf.diag(In)))
        return adj_normalized

    def nomalize_adj(self, adj):
        D = tf.reduce_sum(adj, 1)
        D1 = tf.diag(tf.pow(D, tf.constant(-0.5, shape=[D.shape[0]])))
        return tf.matmul(tf.matmul(D1, adj), D1)

