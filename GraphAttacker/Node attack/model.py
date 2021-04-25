from layers import *
from metrics import *
flags = tf.app.flags
FLAGS = flags.FLAGS

##子图攻击模型
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.vars_dis = {}
        self.vars_gcn2 = {}
        self.vars_gen = {}
        self.placeholders = {}


        self.layers = []
        self.activations = []
        self.activations_gcn2 = []
        self.activations_f = []
        self.rate = 0.01
        self.X = tf.placeholder(tf.float32, [None,20736])
        self.inputs = None
        self.outputs = None
        self.outputs_f =None
        self.out = None
        self.out1 = None
        self.loss = 0
        self.loss_gcn2 = 0
        self.D_loss = 0
        self.G_loss = 0
        self.C_loss = 0
        self.accuracy = 0
        self.accuracy_gcn2 = 0
        self.accuracy_f = 0
        self.optimizer = None
        self.opt_op = None
        self.opt_op_gcn2 = None
        self.MAG_op = None
        self.SD_op = None
        self.AD_op = None

        self.lables=None

        self.lianxu_or_lisan = None

    def _build(self):
        raise NotImplementedError

    def _build_dis(self):
        raise NotImplementedError

    def _build_gen(self):
        raise NotImplementedError

    def _build_gcn2(self):
        raise NotImplementedError

    def _build_gcn3(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope('gcn1'):
            self._build()
        with tf.variable_scope('gcn2'):
            self._build_gcn2()
        with tf.variable_scope('discriminator'):
            self._build_dis()
        with tf.variable_scope('gen'):
            self._build_gen()

        # Build sequential layer model
        #gcn1

        self.lables = self.placeholders['labels']

        self.activations.append(self.inputs)
        hidden = self.layers1(self.activations[-1])
        self.activations.append(hidden)
        self.out = hidden
        hidden = self.layers2(self.activations[-1])
        self.out1 = hidden
        self.activations.append(hidden)
        hidden = self.layers3(self.activations[-1])

        self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.prediction = tf.matmul(self.out1, self.gen_variable['weight']) + self.gen_variable['bias']    #子图gcn提取的低维特征*生成器G的扩维矩阵得到生成子图
        self.prediction_lianxu = (tf.transpose(self.prediction) + self.prediction)/2                       #转置相加再/2，对称矩阵
        self.prediction_test = tf.sigmoid(self.prediction_lianxu)
        self.prediction1 = tf.sign(tf.nn.relu(self.prediction_lianxu))
        self.prediction_MAG = tf.reshape(self.prediction_test,[-1,self.dim*self.dim])                        #把生成子图展开成一维给G训练，用连续数值的生成子图

        self.test = self.replace_sim(self.prediction_test, self.ori_adj, self.K_num, self.sub_num)
        self.prediction_AD_lisan = tf.sign(tf.nn.relu(self.test - 0.5))                                     #最终生成的子图，离散的
        self.prediction_SD = tf.reshape(self.prediction_AD_lisan, [-1, self.dim * self.dim])                 #把生成子图展开成一维给SD训练，用离散数值的生成子图
        self.prediction_AD = [self.test]                                                                    #连续数值的生成子图给攻击器AD训练

        ###gcn2   创建函数gcn的卷积层
        self.activations_gcn2.append(self.inputs)
        hidden_gcn2 = self.layers4(self.activations_gcn2[-1])

        self.activations_gcn2.append(hidden_gcn2)
        hidden_gcn2 = self.layers5(self.activations_gcn2[-1])

        self.activations_gcn2.append(hidden_gcn2)
        hidden_gcn2 = self.layers6(self.activations_gcn2[-1])

        self.activations_gcn2.append(hidden_gcn2)
        self.outputs_gcn2 = self.activations_gcn2[-1]

        variables_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars_all = {var.name: var for var in variables_all}


        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gcn1')
        self.vars = {var.name: var for var in variables}

        variables_gcn2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gcn2')
        self.vars_gcn2 = {var.name: var for var in variables_gcn2}


        variables_dis = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        self.vars_dis = {var.name: var for var in variables_dis}

        variables_gen = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')
        self.vars_gen = {var.name: var for var in variables_gen}

        self.vars_all1 = {var.name: var for var in variables or variables_gcn2}

        self.t = tf.argmax(self.outputs, 1)

        #得到函数gcn的参数，对生成子图进行gcn运算
        self.activations_f.append(self.inputs)
        self.var_l1 = self.layers4.vars['weights_0']
        self.var_l2 = self.layers5.vars['weights_0']
        self.var_l3 = self.layers6.vars['weights_0']
        self.attack_node = self.placeholders['attack_node']

        self.outputs_f,self.attack_outputs = self.gcn(self.prediction_AD,self.activations_f,self.var_l1,self.var_l2,self.var_l3,self.num_features_nonzero,0.5,self.attack_node)
        self.outputs_f_AD, self.attack_outputs_AD = self.gcn([self.prediction_AD_lisan], self.activations_f, self.var_l1, self.var_l2,
                                                       self.var_l3, self.num_features_nonzero, 0.5, self.attack_node)
        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, var_list=self.vars)
        self.opt_op_gcn2 = self.optimizer.minimize(self.loss_gcn2, var_list=self.vars_gcn2)

        self.SD_op = tf.train.AdamOptimizer().minimize(self.SD_loss, var_list=self.vars_dis)  # 判别器的训练器
        self.MAG_op = tf.train.AdamOptimizer(learning_rate=0.03).minimize(self.MAG_loss, var_list=self.vars_gen)  # 生成器的训练器
        self.AD_op = tf.train.AdamOptimizer(learning_rate=0.035).minimize(self.AD_loss, var_list=self.vars_gen)  # 生成器的训练器


    #根据策略不同，选择修改的连边范围，例如3阶子图攻击中，生成的子图是三阶子图全范围，用这个函数限定只将某些节点连边的修改应用到原子图中
    def replace(self,gen_adj,ori_adj,num):
        adj_num = gen_adj.shape[0].value - num
        gen_adj_r0,gen_adj_r1 = tf.split(gen_adj,[num,adj_num],0)
        ori_adj_r0,ori_adj_r1 = tf.split(ori_adj,[num,adj_num],0)
        gen_adj_c0, gen_adj_c1 = tf.split(gen_adj, [num, adj_num], 1)
        # ori_adj_c0, ori_adj_c1 = tf.split(ori_adj, [num, adj_num], 1)
        adj = tf.concat([gen_adj_r0,ori_adj_r1],0)
        adj_c0, adj_c1 = tf.split(adj, [num, adj_num], 1)
        adj_finnal = tf.concat([gen_adj_c0,adj_c1],1)
        return adj_finnal

    def replace_sim(self,gen_adj,ori_adj,num, sub_num):
        adj_num = gen_adj.shape[0].value - sub_num
        gen_adj_r0, gen_adj_r1 = tf.split(gen_adj, [sub_num, adj_num], 0)
        ori_adj_r0, ori_adj_r1 = tf.split(ori_adj, [sub_num, adj_num], 0)
        gen_adj_c0, gen_adj_c1 = tf.split(gen_adj, [sub_num, adj_num], 1)
        adj = tf.concat([ori_adj_r0,gen_adj_r1], 0)
        adj_c0, adj_c1 = tf.split(adj, [sub_num, adj_num], 1)
        adj1 = tf.concat([adj_c0,gen_adj_c1], 1)

        adj_num1 = gen_adj.shape[0].value - num
        gen1_adj_r0,gen1_adj_r1 = tf.split(adj1,[num,adj_num1],0)
        ori1_adj_r0,ori1_adj_r1 = tf.split(ori_adj,[num,adj_num1],0)
        gen1_adj_c0, gen1_adj_c1 = tf.split(adj1, [num, adj_num1], 1)
        # ori_adj_c0, ori_adj_c1 = tf.split(ori_adj, [num, adj_num], 1)
        adj2 = tf.concat([gen1_adj_r0,ori1_adj_r1],0)
        adj2_c0, adj2_c1 = tf.split(adj2, [num, adj_num1], 1)
        adj_finnal = tf.concat([gen1_adj_c0,adj2_c1],1)
        return adj_finnal

    #gcn作为函数，计算生成网络的预测结果
    def gcn(self,support,X,w1,w2,w3,input_dim,dropout,attack_node):

        self.activations_f.append(self.inputs)
        self.var_l1 = self.layers4.vars['weights_0']
        input_gcn = sparse_dropout(X[-1], dropout, input_dim)
        supports1 = list()
        pre_sup = tf.sparse_tensor_dense_matmul(input_gcn, w1)
        In = tf.constant(1, shape=[support[0].shape[0]], dtype=tf.float32)
        support1, D, D1 = self.nomalize_adj(tf.add(support[0], tf.diag(In)))
        support1_ = tf.matmul(support1, pre_sup)
        supports1.append(support1_)
        output1 = tf.add_n(supports1)
        gcn1 = tf.nn.relu(output1)
        X.append(gcn1)

        hidden1_gcn = tf.nn.dropout(X[-1], dropout )
        pre_sup_2 = tf.matmul(hidden1_gcn, w2)
        support_dot2 = tf.matmul(support1, pre_sup_2)
        output2 = support_dot2
        gcn2 = tf.nn.relu(output2)
        X.append(gcn2)

        hidden2_gcn = tf.nn.dropout(X[-1], dropout )
        pre_sup_3 = tf.matmul(hidden2_gcn, w3)
        support_dot3 = tf.matmul(support1, pre_sup_3)
        gcn3 = support_dot3
        X.append(gcn3)
        outputs_f = X[-1]
        attack_outputs = outputs_f * attack_node
        return outputs_f,attack_outputs

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


    # 二分类判别器
    def discriminator(self,x):
        SD_h1 = tf.nn.relu(tf.matmul(x, self.dis_variable['D_w1']) + self.dis_variable['D_b1'])  # 输入乘以D_W1矩阵加上偏置D_b1，D_h1维度为[N, 128]
        SD_logit = tf.matmul(SD_h1, self.dis_variable['D_w2']) + self.dis_variable['D_b2']  # D_h1乘以D_W2矩阵加上偏置D_b2，D_logit维度为[N, 1]
        SD_prob = tf.nn.sigmoid(SD_logit)  # D_logit经过一个sigmoid函数，D_prob维度为[N, 1]

        return SD_prob, SD_logit  # 返回D_prob, D_logit

    def nomalize_adj(self,adj):
        D=tf.reduce_sum(adj,1)
        D1=tf.diag(tf.pow(D,tf.constant(-0.5,shape=[D.shape[0]])))
        return tf.matmul(tf.matmul(D1,adj),D1) ,D ,D1




class GraphAttacker(Model):
    def __init__(self, placeholders, input_dim, dim, K_num, sub_num, **kwargs):
        super(GraphAttacker, self).__init__(**kwargs)
        self.dim = dim
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.ori_adj = placeholders['support'][0]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.num_hidden = 64  #256
        self.X = tf.reshape(self.placeholders['support'],[-1,self.dim*self.dim])  #7333264   20736
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.K_num = K_num
        self.sub_num = sub_num
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers1.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var_gcn2 in self.layers4.vars.values():
            self.loss_gcn2 += FLAGS.weight_decay * tf.nn.l2_loss(var_gcn2)


        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        self.loss_gcn2 += masked_softmax_cross_entropy(self.outputs_gcn2, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

##原Dloss
        self.SD_real, self.SD_logit_real = self.discriminator(self.X)  # 取得判别器判别的真实手写数字的结果
        self.SD_fake, self.SD_logit_fake = self.discriminator(self.prediction_SD)  # 取得判别器判别的生成的手写数字的结

        self.SD_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.SD_logit_real, labels=tf.ones_like(
            self.SD_logit_real)))  # 对判别器对真实样本的判别结果计算误差(将结果与1比较)
        self.SD_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.SD_logit_fake, labels=tf.zeros_like(
            self.SD_logit_fake)))  # 对判别器对虚假样本(即生成器生成的手写数字)的判别结果计算误差(将结果与0比较)


        self.MAG_loss = self.rate * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.SD_logit_fake,
                                                                                         labels=tf.ones_like(self.SD_logit_fake))) \
                      + tf.reduce_mean(tf.square(self.X - self.prediction_MAG))  # 生成器的误差(将判别器返回的对虚假样本的判别结果与1比较)



        self.SD_loss = self.SD_loss_real + self.SD_loss_fake  # 判别器的误差

        # self.G1_loss = masked_softmax_cross_entropy(self.outputs_f, self.placeholders['labels'],
        #                                       self.placeholders['labels_mask'])

        self.AD_loss = masked_softmax_cross_entropy(self.attack_outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.accuracy_gcn2 = masked_accuracy(self.outputs_gcn2, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.accuracy_attack = masked_accuracy(self.attack_outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.accuracy_f = masked_accuracy(self.outputs_f, self.placeholders['labels'],
                                          self.placeholders['labels_mask'])
        self.accuracy_attack_AD_lisan = masked_accuracy(self.attack_outputs_AD, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])


    def _build(self):

        #子图gcn，用来提取子图的低维特征
        self.layers1=(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers2=(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            # sparse_inputs=True,
                                            logging=self.logging))

        self.layers3=(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
    def _build_gcn2(self):

        #函数gcn，用来在model内计算生成子图的预测结果
        self.layers4=(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers5=(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            # sparse_inputs=True,
                                            logging=self.logging))

        self.layers6=(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))



    def _build_dis(self):
        self.dis_variable = {
            'D_w1': tf.Variable(tf.random_normal([self.dim*self.dim, self.num_hidden])),
            'D_b1': tf.Variable(tf.random_normal([self.num_hidden])),
            'D_w2': tf.Variable(tf.random_normal([self.num_hidden, 1])),
            'D_b2': tf.Variable(tf.random_normal([1])),
        }

    def _build_gen(self):
        self.gen_variable = {
            'weight': tf.Variable(tf.random_normal([FLAGS.hidden1, self.dim])),
            'bias' : tf.Variable(tf.random_normal([self.dim, self.dim])),
        }

    def predict(self):
        return tf.nn.softmax(self.outputs)

    def __get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
        return (tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -
                             (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon)))

    def cross_entropy(preds, labels):
        """Softmax cross-entropy loss with masking."""
        loss = labels * -math.log(preds) + (1 - labels) * -math.log(1 - preds)

        return tf.reduce_mean(loss)

##完整网络的gcn
class GCNModel(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.var_l1 = self.layers[0].vars['weights_0']
        self.var_l2 = self.layers[1].vars['weights_0']
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        self.t = tf.argmax(self.outputs, 1)
        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(GCNModel):
    def __init__(self, placeholders, input_dim,  **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        # self.sd_loss = reconstructions

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution_all(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution_all(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))



    def predict(self):
        return tf.nn.softmax(self.outputs)


def gcn1(feature,support,var1,var2, dropout, num_features_nonzero):
    activations= []
    activations.append(feature)
    x1 = activations[-1]
    x1 = sparse_dropout(x1, 1 - dropout, num_features_nonzero)
    pre_sup_1 = tf.sparse_tensor_dense_matmul(x1, var1)
    support_1 = tf.sparse_tensor_dense_matmul(support, pre_sup_1)
    gcn1 = tf.nn.relu(support_1)
    activations.append(gcn1)

    x2 = activations[-1]
    x2 = tf.nn.dropout(x2, 1 - dropout)
    pre_sup_2 = tf.matmul(x2, var2)
    support_2 = tf.sparse_tensor_dense_matmul(support, pre_sup_2)
    gcn2 = tf.nn.relu(support_2)
    activations.append(gcn2)
    output = activations[-1]
    return output
