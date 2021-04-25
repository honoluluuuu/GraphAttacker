from layers import *
from utils import *
import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


#versione base del modello di neural network, ripreso dalla definizione usata in Keras
#sarebbe una classe astratta, molti metodi vanno implementati e alcune variabili inizializzate
class AttackNet(object):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__.lower()
        self.weights = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.activations_f = []
        self.activations_pool_f=[]
        self.inputs = None
        self.output_ori = None
        self.loss = 0
        self.accuracy_att = 0
        self.accuracy_ori = 0
        self.optimizer = None
        self.opt_op = None
        self.out = []
        self.output = None

        self.activations_ori = []
        self.rate = 0.01
    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
        with tf.variable_scope('discriminator'):
            self._build_dis()
        with tf.variable_scope('gen'):
            self._build_gen()
# costruzione del modello con layers generici

        # for i in range(len(self.support)):
        self.activations.append(self.inputs)
        self.d_fea = self.diffpool.gcn_layers[0](self.inputs, self.support, self.num_features_nonzero)
        self.gen_adj = tf.matmul(self.d_fea, self.gen_variable['weight']) + self.gen_variable['bias'] #self.diffpool.d_fea
        self.gen_adj = (tf.transpose(self.gen_adj) + self.gen_adj)/2                       #转置相加再/2，对称矩阵
        self.gen_adj = tf.sigmoid(self.gen_adj)
        self.gen_adj = self.replace(self.gen_adj,self.adj, self.att_size)

        self.gen_adj_G = tf.reshape(self.gen_adj, [-1, self.num_nodes * self.num_nodes])

        self.gen_adj_L = tf.sign(tf.nn.relu(self.gen_adj - 0.5))                                     #最终生成的子图，离散的
        self.gen_adj_D = tf.reshape(self.gen_adj_L, [-1, self.num_nodes * self.num_nodes])
        # self.pre = self.prediction1(self.adj,self.inputs, self.pool_mat, self.diffpool.gcn_layers, self.diffpool.pool_layers,
        #                            self.num_features_nonzero,self.dropout)

        # #
        self.pre, self.w0, self.w1, self.w2, self.w3= self.diffpool.predict(self.gen_adj_L, self.inputs, self.pool_mat, self.num_features_nonzero, self.num_nodes, self.assign_dim, self.hidden_dim, self.embedding_dim,
                                         self.nomalize_adj, self.diffpool.gcn_layers, self.diffpool.pool_layers, self.dropout)

        self.output_ori,self.v0, self.v1, self.v2, self.v3 = self.diffpool.predict(self.adj, self.inputs, self.pool_mat, self.num_features_nonzero, self.num_nodes, self.assign_dim, self.hidden_dim, self.embedding_dim,
                                         self.nomalize_adj, self.diffpool.gcn_layers, self.diffpool.pool_layers, self.dropout)

        # salvo per comodità le variabili del modello invece che tenerle solo in tf.GraphKeys.GlOBALVARIABLES
        self.weights = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)}

        variables_dis = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        self.vars_dis = {var.name: var for var in variables_dis}

        variables_gen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')
        self.vars_gen = {var.name: var for var in variables_gen}
# inizializzo loss e accuracy

        self._loss()
        self._accuracy()

        self.class_ori = tf.argmax(self.output_ori, 1)
        self.class_att = tf.argmax(self.pre, 1)
        # self.opt_op = self.optimizer.minimize(self.loss)
        self.D_op = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.vars_dis)  # 判别器的训练器
        self.G_op = tf.train.AdamOptimizer(learning_rate=0.03).minimize(self.G_loss, var_list=self.vars_gen)  # 生成器的训练器
        self.C_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.C_loss, var_list=self.vars_gen)  # 生成器的训练器




    def nomalize_adj(self,adj,dim):
        D=tf.reduce_sum(adj,1)
        D1=tf.diag(tf.pow(D,tf.constant(-0.5,shape=[dim])))
        return tf.matmul(tf.matmul(D1,adj),D1)

    def replace(self, gen_adj, ori_adj, dim):
        o_r0, o_r1 = tf.split(ori_adj,[dim, tf.shape(ori_adj)[0] - dim], 0)
        o_c0, o_c1 = tf.split(o_r0, [dim, tf.shape(ori_adj)[0] - dim], 1)

        g_r0, g_r1 = tf.split(gen_adj,[dim, tf.shape(gen_adj)[0] - dim], 0)
        g_c0, g_c1 = tf.split(g_r0, [dim, tf.shape(gen_adj)[0] - dim], 1)


        adj_r = tf.concat([g_c0,o_c1],1)
        adj = tf.concat([adj_r,o_r1],0)
        return adj


    def _loss(self):
        raise NotImplementedError

    def discriminator(self,x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.dis_variable['D_w1']) + self.dis_variable['D_b1'])  # 输入乘以D_W1矩阵加上偏置D_b1，D_h1维度为[N, 128]
        D_logit = tf.matmul(D_h1, self.dis_variable['D_w2']) + self.dis_variable['D_b2']  # D_h1乘以D_W2矩阵加上偏置D_b2，D_logit维度为[N, 1]
        D_prob = tf.nn.sigmoid(D_logit)  # D_logit经过一个sigmoid函数，D_prob维度为[N, 1]

        return D_prob, D_logit  # 返回D_prob, D_logit

    def _accuracy(self):
        raise NotImplementedError

class GraphAttack(AttackNet):
    def __init__(self, placeholders,diffpool, input_dim, featureless,idx, num_graphs, num_nodes,att_size, assign_dim, with_pooling, **kwargs):
        super(GraphAttack, self).__init__(**kwargs)
        #######
        self.num_hidden = 64
        self.hidden_dim = 64
        self.diffpool = diffpool
        self.att_size = att_size
        self.dropout = placeholders['dropout']
        self.adj = placeholders['att_adj']
        self.fea = placeholders['att_fea']
        self.assign_dim = assign_dim
        self.embedding_dim = 2

        #######
        self.pooling = with_pooling
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.support = placeholders['support']
        self.adj_G = tf.reshape(self.adj, [-1, self.num_nodes * self.num_nodes])
        self.adj_G = tf.cast(self.adj_G, dtype=tf.float32)

        self.idx =  idx#placeholders['idx'] #idx#
        # self.batch = placeholders['batch']
        self.inputs = placeholders['att_fea']
        self.pool_mat = placeholders['pool_mat']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.placeholders = placeholders
        self.featureless = featureless
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # # Weight decay loss
        # for var in self.layers[0].weights.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        ##cross entropy loss dopo aver applicato un softmax layer
        # self.loss += masked_cross_entropy(self.output, self.placeholders['labels'],
        #                                           self.placeholders['labels_mask'])
        self.D_real, self.D_logit_real = self.discriminator(self.adj_G)  # 取得判别器判别的真实手写数字的结果
        self.D_fake, self.D_logit_fake = self.discriminator(self.gen_adj_D)  # 取得判别器判别的生成的手写数字的结
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real, labels=tf.ones_like(
            self.D_logit_real)))  # 对判别器对真实样本的判别结果计算误差(将结果与1比较)
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.zeros_like(
            self.D_logit_fake)))  # 对判别器对虚假样本(即生成器生成的手写数字)的判别结果计算误差(将结果与0比较)

        self.G_loss = self.rate * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake,
                                                                                         labels=tf.ones_like(self.D_logit_fake))) \
                      + tf.reduce_mean(tf.square(self.adj_G - self.gen_adj_G))  # 生成器的误差(将判别器返回的对虚假样本的判别结果与1比较)



        self.D_loss = self.D_loss_real + self.D_loss_fake  # 判别器的误差


        self.C_loss = masked_cross_entropy(self.pre, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])  # -1 *



    #
    def _accuracy(self):
        self.accuracy_att = masked_accuracy(self.pre, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.accuracy_ori = masked_accuracy(self.output_ori, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(ConvolutionalLayer(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            activation=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            featureless = self.featureless))


        self.layers.append(ConvolutionalLayer(input_dim=FLAGS.hidden2,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            activation=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=False,
                                            featureless = False))

    def _build_dis(self):
        self.dis_variable = {
            'D_w1': tf.Variable(tf.random_normal([self.num_nodes * self.num_nodes, self.num_hidden])),
            'D_b1': tf.Variable(tf.random_normal([self.num_hidden])),
            'D_w2': tf.Variable(tf.random_normal([self.num_hidden, 1])),
            'D_b2': tf.Variable(tf.random_normal([1])),
        }

    def _build_gen(self):
        self.gen_variable = {
            'weight': tf.Variable(tf.random_normal([FLAGS.hidden2, self.num_nodes])),
            'bias': tf.Variable(tf.random_normal([self.num_nodes, self.num_nodes]))}
    # def predict(self):
        # return tf.nn.softmax(self.output)







class BaseNet(object):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__.lower()
        self.weights = {}
        self.placeholders = {}
        self.gcn_layers = []
        self.layers = []
        self.pool_layers = []
        self.activations = []
        self.activations_f = []
        self.activations_pool = []
        self.activations_pool_f = []
        self.inputs = None
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.out = []
        self.out_f = []
        self.output = None
        self.output_f = None
        self.d_fea = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
# costruzione del modello con layers generici

        # 单个图

        # self.activations_f.append(self.inputs[i])
        # self.activations_pool[i].append(self.inputs[i])
        #
        # 第一个gcn块
        self.activations_f.append(self.inputs_att)
        self.activations_pool_f.append(self.inputs_att)
        out_all_f = []
        hidden1 = self.gcn_layers[0](self.activations_f[-1], self.att_adj, self.att_num_features_nonzero)  # num_nodes*hidden_dim
        self.x_all_f = [hidden1]
        self.activations_f.append(hidden1)

        hidden2 = self.gcn_layers[1](self.activations_f[-1], self.att_adj, self.att_num_features_nonzero)  # num_nodes*hidden_dim
        self.x_all_f.append(hidden2)
        self.activations_f.append(hidden2)

        hidden3 = self.gcn_layers[2](self.activations_f[-1], self.att_adj, self.att_num_features_nonzero)  # num_nodes*embedding_dim
        self.x_all_f.append(hidden3)

        self.x_tensor_f = tf.multiply(tf.concat(self.x_all_f, axis=1), self.att_pool_mat)
        self.out_1_f = tf.reduce_max(self.x_tensor_f, 1)
        out_all_f.append(self.out_1_f)

        # 第一个pool块
        self.assign_tensor_f = self.pool_layers[0](self.activations_pool_f[-1], self.att_adj,
                                                   self.att_num_features_nonzero)
        self.pool_all_f = [self.assign_tensor_f]
        self.activations_pool_f.append(self.assign_tensor_f)
        self.assign_tensor_f = self.pool_layers[1](self.activations_pool_f[-1], self.att_adj,
                                                   self.att_num_features_nonzero)
        self.pool_all_f.append(self.assign_tensor_f)
        self.activations_pool_f.append(self.assign_tensor_f)
        self.assign_tensor_f = self.pool_layers[2](self.activations_pool_f[-1], self.att_adj,
                                                   self.att_num_features_nonzero)
        self.pool_all_f.append(self.assign_tensor_f)

        self.assign_tensor_f = tf.nn.softmax(
            tf.layers.dense(tf.concat(self.pool_all_f, axis=1), self.assign_dim, name='pool'), axis=-1)

        self.x_f = tf.matmul(tf.transpose(self.assign_tensor_f), self.x_tensor_f)
        self.assign_a_f = tf.matmul(tf.matmul(tf.transpose(self.assign_tensor_f), self.A), self.assign_tensor_f)

        # 第二个gcn块
        self.activations_f.append(self.x_f)
        hidden21 = self.gcn_layers[3](self.activations_f[-1], self.assign_a_f, self.att_num_features_nonzero)
        self.x_all_2_f = [hidden21]
        self.activations_f.append(hidden21)

        hidden22 = self.gcn_layers[4](self.activations_f[-1], self.assign_a_f, self.att_num_features_nonzero)
        self.x_all_2_f.append(hidden22)
        self.activations_f.append(hidden22)

        hidden23 = self.gcn_layers[5](self.activations_f[-1], self.assign_a_f, self.att_num_features_nonzero)
        self.x_all_2_f.append(hidden23)
        self.x_tensor2_f = tf.concat(self.x_all_2_f, axis=1)
        self.out_2_f = tf.reduce_max(self.x_tensor2_f, 1)
        out_all_f.append(self.out_2_f)
        output = tf.expand_dims(tf.concat(out_all_f, axis=0), 0)
        pre_0 = tf.nn.relu(tf.layers.dense(output, self.hidden_dim, name='gcn1'))
        self.output_att = tf.layers.dense(pre_0, self.embedding_dim, name='gcn2')  # 输出

        self.output_att1 = self.predict(self.A, self.inputs_att, self.att_pool_mat, self.att_num_features_nonzero, self.num_nodes, self.assign_dim, self.hidden_dim,
                                        self.embedding_dim, self.nomalize_adj, self.gcn_layers, self.pool_layers, self.dropout)


        for i in range(len(self.support)):

            out_all = []
            self.activations.append([])
            self.activations_pool.append([])

            self.activations[i].append(self.inputs[i])
            self.activations_pool[i].append(self.inputs[i])

            #第一个gcn块
            hidden1 = self.gcn_layers[0](self.inputs[i], self.support[i], self.num_features_nonzero[i])
            x_all = [hidden1]
            self.activations[i].append(hidden1)

            hidden2 = self.gcn_layers[1](hidden1, self.support[i], self.num_features_nonzero[i])
            x_all.append(hidden2)
            self.activations[i].append(hidden2)

            hidden3 = self.gcn_layers[2](hidden2, self.support[i], self.num_features_nonzero[i])
            x_all.append(hidden3)
            # self.d_fea = hidden1
            # self.d_fea2 = hidden2


            self.x_tensor = tf.multiply(tf.concat(x_all,axis=1),self.pool_mat1[i])
            self.out_1 = tf.reduce_max(self.x_tensor ,1)
            out_all.append(self.out_1)

            # 第一个pool块
            self.assign_tensor =self.pool_layers[0](self.activations_pool[i][-1], self.support[i], self.num_features_nonzero[i])
            self.pool_all = [self.assign_tensor]
            self.activations_pool[i].append(self.assign_tensor)
            self.assign_tensor = self.pool_layers[1](self.activations_pool[i][-1], self.support[i],
                                                     self.num_features_nonzero[i])
            self.pool_all.append(self.assign_tensor)
            self.activations_pool[i].append(self.assign_tensor)
            self.assign_tensor = self.pool_layers[2](self.activations_pool[i][-1], self.support[i],
                                                     self.num_features_nonzero[i])
            self.pool_all.append(self.assign_tensor)

            self.assign_tensor = tf.nn.softmax(tf.layers.dense(tf.concat(self.pool_all,axis=1), self.assign_dim,name='pool',reuse=True),axis=-1)


            self.x =tf.matmul(tf.transpose(self.assign_tensor),self.x_tensor)
            self.assign_a = tf.matmul(tf.matmul(tf.transpose(self.assign_tensor),self.adj[i]),self.assign_tensor)

            # 第二个gcn块
            self.activations[i].append(self.x)
            hidden21 = self.gcn_layers[3](self.activations[i][-1], self.assign_a, self.num_features_nonzero[i])
            self.x_all_2 = [hidden21]
            self.activations[i].append(hidden21)

            hidden22 = self.gcn_layers[4](self.activations[i][-1], self.assign_a, self.num_features_nonzero[i])
            self.x_all_2.append(hidden22)
            self.activations[i].append(hidden22)

            hidden23 = self.gcn_layers[5](self.activations[i][-1], self.assign_a, self.num_features_nonzero[i])
            self.x_all_2.append(hidden23)
            self.x_tensor2 = tf.concat(self.x_all_2,axis=1)
            self.out_2 = tf.reduce_max(self.x_tensor2, 1)
            out_all.append(self.out_2)
            output = tf.expand_dims(tf.concat(out_all, axis=0),0)
            pre_0 = tf.nn.relu(tf.layers.dense(output,self.hidden_dim,name='gcn1',reuse=True))

            self.pre = tf.layers.dense(pre_0,self.embedding_dim,name='gcn2',reuse=True)      #输出
            #
            self.out.append(self.pre)



        for i in range(len(self.support)-1):
            if i == 0:
                self.output = tf.concat([self.out[0],self.out[1]],0)
                # self.output_f = tf.concat([self.out_f[0], self.out_f[1]], 0)
            else:
                self.output = tf.concat([self.output,self.out[i+1]],0)
                # self.output_f = tf.concat([self.output_f, self.out_f[i + 1]], 0)




        self.class_graph = tf.argmax(self.output_att, 1)


# salvo per comodità le variabili del modello invece che tenerle solo in tf.GraphKeys.GlOBALVARIABLES
        self.weights = {var.name: var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)}

# inizializzo loss e accuracy
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self, adj, input, pool_mat, num_features_nonzero, num_nodes, assign_dim, hidden_dim, embedding_dim, nomalize_adj, gcn_layers, pool_layers, dropout):
        In = tf.constant(1, shape=[num_nodes], dtype=tf.float32)
        support1 = nomalize_adj(tf.add(adj, tf.diag(In)), num_nodes)  # 规范化的A
        out_all = []
        # self.activations.append(self.inputs)
        #  第一个gcn块
        input_gcn_1_0 = sparse_dropout(input, 1 - dropout, num_features_nonzero)
        pre_sup = tf.sparse_tensor_dense_matmul(input_gcn_1_0,  gcn_layers[0].weights['weights_0'])
        output1 = tf.nn.relu(tf.matmul(support1, pre_sup) +  gcn_layers[0].weights['bias'])
        x_all = [output1]

        input_1_1 = tf.nn.dropout(output1, 1 - dropout)
        pre_sup_2 = tf.matmul(input_1_1,  gcn_layers[1].weights['weights_0'])
        output2 = tf.nn.relu(tf.matmul(support1, pre_sup_2) +  gcn_layers[1].weights['bias'])
        x_all.append(output2)

        input_1_2 = tf.nn.dropout(output2, 1 - dropout)
        pre_sup_3 = tf.matmul(input_1_2,  gcn_layers[2].weights['weights_0'])
        output3 = tf.matmul(support1, pre_sup_3) +  gcn_layers[2].weights['bias']
        x_all.append(output3)

        x_tensor_1 = tf.multiply(tf.concat(x_all, axis=1), pool_mat)
        out_1 = tf.reduce_max(x_tensor_1, 1)
        out_all.append(out_1)

        # pool 块
        input_gcn_2_0 = sparse_dropout(input, 1 - dropout, num_features_nonzero)
        pre_sup_pool_0 = tf.sparse_tensor_dense_matmul(input_gcn_2_0, pool_layers[0].weights['weights_0'])
        output_pool0 = tf.nn.relu(tf.matmul(support1, pre_sup_pool_0) + pool_layers[0].weights['bias'])
        pool_all = [output_pool0]

        input_2_1 = tf.nn.dropout(output_pool0, 1 - dropout)
        pre_sup_pool_1 = tf.matmul(input_2_1, pool_layers[1].weights['weights_0'])
        output_pool_1 = tf.nn.relu(tf.matmul(support1, pre_sup_pool_1) + pool_layers[1].weights['bias'])
        pool_all.append(output_pool_1)

        input_2_2 = tf.nn.dropout(output_pool_1, 1 - dropout)
        pre_sup_pool_2 = tf.matmul(input_2_2, pool_layers[2].weights['weights_0'])
        output_pool_2 = tf.matmul(support1, pre_sup_pool_2) + pool_layers[2].weights['bias']
        pool_all.append(output_pool_2)
        assign_tensor = tf.nn.softmax(tf.layers.dense(tf.concat(pool_all, axis=1), assign_dim,name='pool', reuse=True), axis=-1)

        assign_x = tf.matmul(tf.transpose(assign_tensor), x_tensor_1)
        assign_a = tf.matmul(tf.matmul(tf.transpose(assign_tensor), adj), assign_tensor)

        # 第二个gcn
        In_1 = tf.constant(1, shape=[assign_dim], dtype=tf.float32)
        support1_1 = nomalize_adj(tf.add(assign_a, tf.diag(In_1)), assign_dim)

        input_3_0 = tf.nn.dropout(assign_x, 1 - dropout)
        pre_sup_gcn_0 = tf.matmul(input_3_0,  gcn_layers[3].weights['weights_0'])
        output_gcn_0 = tf.nn.relu(tf.matmul(support1_1, pre_sup_gcn_0) +  gcn_layers[3].weights['bias'])
        x_all_gcn = [output_gcn_0]

        input_3_1 = tf.nn.dropout(output_gcn_0, 1 - dropout)
        pre_sup_gcn_1 = tf.matmul(input_3_1,  gcn_layers[4].weights['weights_0'])
        output_gcn_1 = tf.nn.relu(tf.matmul(support1_1, pre_sup_gcn_1) +  gcn_layers[4].weights['bias'])
        x_all_gcn.append(output_gcn_1)

        input_3_2 = tf.nn.dropout(output_gcn_1, 1 - dropout)
        pre_sup_gcn_2 = tf.matmul(input_3_2,  gcn_layers[5].weights['weights_0'])
        output_gcn_2 = tf.matmul(support1_1, pre_sup_gcn_2) +  gcn_layers[5].weights['bias']
        x_all_gcn.append(output_gcn_2)

        # x_tensor_2 = tf.multiply(tf.concat(x_all_gcn, axis=1), pool_mat)
        x_tensor_2 = tf.concat(x_all_gcn, axis=1)
        out_2 = tf.reduce_max(x_tensor_2, 1)
        out_all.append(out_2)

        output = tf.expand_dims(tf.concat(out_all, axis=0), 0)
        pre_0 = tf.nn.relu(tf.layers.dense(output, hidden_dim,name='gcn1',reuse=True))
        output_att = tf.layers.dense(pre_0, embedding_dim, name='gcn2', reuse=True)  # 输出

        return output_att, support1,input_gcn_1_0, pre_sup, output1

    #
    def nomalize_adj(self,adj,dim):
        D=tf.reduce_sum(adj,1)
        D1=tf.diag(tf.pow(D,tf.constant(-0.5,shape=[dim])))
        return tf.matmul(tf.matmul(D1,adj),D1)

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError



class GCNGraphs(BaseNet):
    def __init__(self, placeholders, input_dim, featureless,idx, num_graphs, num_nodes, assign_dim, with_pooling, **kwargs):
        super(GCNGraphs, self).__init__(**kwargs)
        #######
        self.num_hidden = 64
        self.hidden_dim = 64
        self.embedding_dim = 2
        self.pred_input_dim = self.num_hidden * 2 + self.embedding_dim
        #######
        self.pooling = with_pooling
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.support = placeholders['support']
        self.adj = placeholders['adj_batch']
        self.att_adj = placeholders['att_adj']
        self.dropout = placeholders['dropout']
        self.num_size = placeholders['num_size']
        self.assign_dim = assign_dim
        # self.support = placeholders['support']
        # self.adj = placeholders['adj_batch']
        # self.att_adj = placeholders['att_support']
        self.A = placeholders['att_A']
        # self.dropout = placeholders['dropout']
        self.idx =  idx#placeholders['idx'] #idx#
        # self.batch = placeholders['batch']
        self.inputs = placeholders['feats']
        self.inputs_att = placeholders['att_fea']
        self.pool_mat1 = placeholders['pool_mat1']
        self.att_pool_mat = placeholders['att_pool_mat']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.att_num_features_nonzero = placeholders['att_num_features_nonzero']
        self.placeholders = placeholders
        self.featureless = featureless
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.gcn_layers[0].weights.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        #cross entropy loss dopo aver applicato un softmax layer
        self.loss += masked_cross_entropy(self.output, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.output, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.accuracy_att = masked_accuracy(self.output_att, self.placeholders['att_labels'],
                                        self.placeholders['att_labels_mask'])
        # self.accuracy_f = masked_accuracy(self.output_f, self.placeholders['labels'],
        #                                 self.placeholders['labels_mask'])

    def _build(self):
        #gcn_1
        self.gcn_layers.append(ConvolutionalLayer(input_dim=self.input_dim,output_dim=self.hidden_dim,placeholders=self.placeholders,activation=tf.nn.relu,
                                              dropout=True,sparse_inputs=True,featureless = self.featureless))
        self.gcn_layers.append(ConvolutionalLayer(input_dim=self.hidden_dim,output_dim=self.hidden_dim,placeholders=self.placeholders,activation=tf.nn.relu,
                                              dropout=True,sparse_inputs=False,featureless = self.featureless))
        self.gcn_layers.append(ConvolutionalLayer(input_dim=self.hidden_dim,output_dim=self.embedding_dim,placeholders=self.placeholders,activation=lambda x: x,
                                              dropout=True,sparse_inputs=False,featureless = self.featureless))

        #pool_1
        self.pool_layers.append(ConvolutionalLayer(input_dim=self.input_dim,output_dim=self.hidden_dim,placeholders=self.placeholders,activation=tf.nn.relu,
                                              dropout=True,sparse_inputs=True,featureless = self.featureless))
        self.pool_layers.append(ConvolutionalLayer(input_dim=self.hidden_dim,output_dim=self.hidden_dim,placeholders=self.placeholders,activation=tf.nn.relu,
                                              dropout=True,sparse_inputs=False,featureless = self.featureless))
        self.pool_layers.append(ConvolutionalLayer(input_dim=self.hidden_dim,output_dim=self.assign_dim,placeholders=self.placeholders,activation=lambda x: x,
                                              dropout=True,sparse_inputs=False,featureless = self.featureless))
        #gcn_2
        self.gcn_layers.append(ConvolutionalLayer_1(input_dim=self.pred_input_dim,output_dim=self.hidden_dim,placeholders=self.placeholders,activation=tf.nn.relu,
                                              dropout=True,sparse_inputs=False,featureless = self.featureless))
        self.gcn_layers.append(ConvolutionalLayer_1(input_dim=self.hidden_dim,output_dim=self.hidden_dim,placeholders=self.placeholders,activation=tf.nn.relu,
                                              dropout=True,sparse_inputs=False,featureless = self.featureless))
        self.gcn_layers.append(ConvolutionalLayer_1(input_dim=self.hidden_dim,output_dim=self.embedding_dim,placeholders=self.placeholders,activation=lambda x: x,
                                              dropout=True,sparse_inputs=False,featureless = self.featureless))



    #
    # def predict(self):
    #     return tf.nn.softmax(self.output)