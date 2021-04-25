from __future__ import division
from __future__ import print_function

from file_utils import *
from utils import *
# from net import GCNGraphs,GraphAttack
from mix_net import GCNGraphs,GraphAttack
# Set random seed
import numpy as np
import copy
import random
import os
seed = 123
budget = 0.4
np.random.seed(seed)
tf.set_random_seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
os.system('nvidia-smi')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True #不全部占满显存, 按需分配
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'PROTEINS', 'which dataset to load')  # PROTEINS
flags.DEFINE_boolean('with_pooling', True, 'whether the mean value for graph labels is computed via pooling(True) or via global nodes(False)')
flags.DEFINE_boolean('featureless', False, 'If nodes are featureless') #only if with_pooling = False
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number; of epochs to train.')
flags.DEFINE_integer('epochs_gan',200, 'Number of epochs to train.')
flags.DEFINE_integer('epochs_d', 10, 'Number of epochs to train.')
flags.DEFINE_integer('epochs_g', 10, 'Number of epochs to train.')
flags.DEFINE_integer('epochs_c', 7, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 16, 'Number of units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

# attack_graph = 0

if FLAGS.dataset=='PROTEINS':
    num_nodes = 43471
    num_graphs = 1113
    tot = 44584
    num_classes = 2
    num_feats = 29
    splits = [[0,7], [7, 7], [7, 10]]
    dataset_name = "proteins"
if FLAGS.dataset=='DD':
    num_nodes = 334925
    num_graphs = 1178
    tot = 44584
    num_classes = 2
    num_feats = 29
    splits = [[0,7], [7, 7], [7, 10]]
    dataset_name = "DD"




# graph, feats_matrix, labels, idx, a_size, feat, graph_size = get_data(num_nodes, num_graphs, num_classes, num_feats, dataset_name)

adj, feature, labels, idx = load_data_basic(num_nodes, num_graphs, num_classes, num_feats, dataset_name)
# adj, labels, idx = load_data_basic_label(num_nodes, num_graphs, num_classes, num_feats, dataset_name)
print("adjacency, attributes and labels parsed")
# path = dataset_name + "/" + dataset_name.upper() + "_node_attributes.txt"
# feats_matrix = np.loadtxt(path, delimiter=',')

path = dataset_name + "/" + dataset_name.upper() + "_node_attributes.txt"
f_l = open("proteins/PROTEINS_node_labels.txt", 'r')
node_l = []
for line in f_l.readlines():
    curLine = line.strip("\n")
    val = int(curLine)
    node_l.append(val)
fea_size = max(node_l)+1
fea_label = np.zeros((num_nodes,fea_size))
for i in range(len(node_l)):
    fea_label[i][node_l[i]]=1
feature = csr_matrix(fea_label)
feats_matrix = fea_label

# f_n = open("proteins/PROTEINS_attack_graph.txt", 'a')
# a_g = []
# num = 0
# while num<99:
#     add = random.randint(0, num_graphs-1)
#     if add not in a_g:
#         f_n.write(str(add)+'\n')
#         num +=1
# f_n.close()

att_g = []
f_n = open("proteins/PROTEINS_attack_graph.txt", 'r')
for line in f_n.readlines():
    curLine = line.strip().split(" ")
    att_g.append(int(curLine[0]))
att_g.sort()




print("normalizzo adj e node attributes")
# support = [preprocess_adj(adj, True, True)]
feat = process_features(feature)
# feat = process_features(feat)
num_supports = 10

input_dim = feats_matrix.shape[1]
label1 = labels.tolist()
ori_A = adj.A
graph, fea, label, ori_adj, a_size, graph_size = single_graph(num_graphs, num_nodes, idx, feats_matrix, ori_A)    #从整个网络中获得单个图的信息


randnum = 123
# graph1, fea1, label1, adj_size1 = data_rand(randnum, graph_cut, fea_cut, adj_size_cut, label_cut)
# graph1, fea1, label1, adj_size1 = data_rand(randnum, graph, feats_matrix, a_size, labels)
graph1, fea1, label1, adj_size1 = data_rand(randnum, graph, fea, a_size, labels)
sum = int(len(graph)/num_supports)*num_supports
support, features, label, pool_mat, batch, adj_batch, num_size, pool_mat1 = data_batch1(sum, num_supports, graph_size, graph1, fea1, label1, adj_size1)


y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask1, val_mask, test_mask = get_splits_graphs_basic(num_supports, label[0], splits[0], splits[1], splits[2], idx[0])


train_mask = np.ones((num_supports),dtype=bool)
train_batch = int(batch * 0.7)
test_batch = batch-train_batch

y_train = copy.deepcopy(label)
y_test = copy.deepcopy(label)
for i in range(len(label)):
    for j in range(y_train[i].shape[0]):
        if not train_mask[j]:
            y_train[i][j][0]=y_train[i][j][1] = 0
        if not test_mask[j]:
            y_test[i][j][0]=y_test[i][j][1] = 0




attack_success = 0  # 该节点成功攻击的次数
gcn_accuracy = 0  # 原GCN对该节点的正确分类次数
attack_num = 0
AML = 0
ASR = 0
start = 0
while(attack_num<100):
    tf.reset_default_graph()

    if attack_num<start:
        attack_num +=1
        print(attack_num,"<",start)
        tf.reset_default_graph()
        continue

    attack_graph = att_g[attack_num]


    att_adj, att_fea, ori_support, att_pooling, ori_class, ori_label, att_label, att_mask, graph_link, att_size, att_fea_dense, graph_fea= att_graph1(
        attack_graph, graph, fea, labels, graph_size, a_size)


    GCN_placeholders = {
        # 'support': [tf.sparse_placeholder(tf.float32) for i in range(num_supports)],
        'support': [tf.placeholder(tf.float32) for i in range(num_supports)],
        'feats': [tf.placeholder(tf.float32) for i in range(num_supports)],
        'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'att_A': tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': [tf.placeholder(tf.int32) for i in range(num_supports)],  # helper variable for sparse dropout
        'pool_mat1': [tf.placeholder(tf.float32, shape=(graph_size, 130)) for i in range(num_supports)],
        'adj_batch': [tf.placeholder(tf.float32) for i in range(num_supports)],
        'att_labels': tf.placeholder(tf.float32, shape=(None, ori_label.shape[1])),
        'att_labels_mask': tf.placeholder(tf.int32),
        # 'att_adj': tf.sparse_placeholder(tf.float32),
        'att_adj': tf.placeholder(tf.float32),
        'att_fea': tf.placeholder(tf.float32),
        'att_pool_mat': tf.placeholder(tf.float32, shape=(graph_size, 130)),
        'att_num_features_nonzero': tf.placeholder(tf.int32),
        'num_size':[tf.placeholder(tf.int64) for i in range(num_supports)],
    }


    featureless = (FLAGS.featureless)
    network = GCNGraphs(GCN_placeholders, input_dim, featureless, adj_size1, num_supports, graph_size, int(graph_size*0.1), FLAGS.with_pooling)

    # Initialize session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())
    all_vars = tf.global_variables()


    train_loss = [0. for i in range(0, FLAGS.epochs)]
    val_loss = [0. for i in range(0, FLAGS.epochs)]

    def evaluate(adj_batch, features, support, pool_mat1, labels, mask, att_adj, att_support,att_fea,ori_label,att_mask,att_pooling,num_size, placeholders):
        t_test = time.time()
        feed_dict_val = build_dictionary_GCN(adj_batch, features, support, pool_mat1, labels, mask,att_support,att_fea,ori_label,att_mask,att_pooling, num_size, GCN_placeholders)
        feed_dict_val.update({GCN_placeholders['att_A']: att_adj})
        outs_val = sess.run([network.loss, network.accuracy, network.accuracy_att], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

    # Train network
    for epoch in range(FLAGS.epochs):
        t = time.time()
        acc = 0
        test_acc= 0
        for i in range(train_batch):
            train_dict = build_dictionary_GCN(adj_batch[i], features[i], support[i], pool_mat1[i], label[i], train_mask, ori_support, att_fea, ori_label, att_mask, att_pooling, num_size[i], GCN_placeholders)
            train_dict.update({GCN_placeholders['dropout']: FLAGS.dropout})
            train_dict.update({GCN_placeholders['att_A']: att_adj})
        # Training step
            train_out = sess.run([network.opt_op, network.loss, network.accuracy,network.output, network.output_att], feed_dict=train_dict)
        #     train_out = sess.run([network.opt_op],feed_dict=train_dict)
            acc +=train_out[2]

        for i in range(test_batch):
            test_dict = build_dictionary_GCN(adj_batch[train_batch+i], features[train_batch+i], support[train_batch+i], pool_mat1[train_batch+i],
                                              label[train_batch+i], train_mask, ori_support, att_fea, ori_label, att_mask,
                                              att_pooling, num_size[train_batch+i], GCN_placeholders)
            test_dict.update({GCN_placeholders['dropout']: FLAGS.dropout})
            test_dict.update({GCN_placeholders['att_A']: att_adj})
            # Training step
            test_out = sess.run(
                [network.loss, network.accuracy, network.output, network.output_att],feed_dict=test_dict)

            # Atest_out = sess.run(
            #     [network.output_att1], feed_dict=test_dict)

            test_acc +=test_out[1]


        acc = acc/train_batch
        test_acc = test_acc/test_batch
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_out[1]), "train_acc=", "{:.5f}".format(acc),"test_acc=", "{:.5f}".format(test_acc))
    print("Optimization Finished!")



    ##########################
    #        Attack
    node_num = graph_size
    modify_num = 0
    att_time = 0
    acc_node = 0
    att_success = 0
    gcn_acc = 0
    save_dir = "result/PROTEINS_result-mix-"+str(budget)+"-onehot0.txt"
    # save_dir = "result/PROTEINS_result-A-lr_c0.033-0.4-label.txt"
    f = open(save_dir, 'a')

    while (att_success == 0 and att_time < 10):
        att_time +=1
        # adj_save = 'PROTEINS/adj_PROTEINS-A-lr_c0.033-0.4-label/gen_PROTEINS_' + str(attack_graph) + '_' + str(att_time)
        adj_save = 'PROTEINS/adj_PROTEINS-mix-'+str(budget)+'-onehot/gen_PROTEINS_' + str(attack_graph) + '_' + str(att_time)
        print("第", attack_graph, "个图的第", att_time , "次攻击")

        Att_placeholders = {
            'support': tf.placeholder(tf.float32),
            'labels': tf.placeholder(tf.float32, shape=(None, ori_label.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
            'pool_mat': tf.placeholder(tf.float32, shape=(graph_size, 130)),
            'att_adj':tf.placeholder(tf.float32),
            # 'att_fea': tf.sparse_placeholder(tf.float32, shape=tf.constant(feat[2], dtype=tf.int64))
            'att_fea': tf.placeholder(tf.float32),
        }
        Attgraph = GraphAttack(Att_placeholders, network, input_dim, featureless, adj_size1, num_supports, graph_size, att_size, int(graph_size*0.1), FLAGS.with_pooling)

        var_all = []
        [var_all.append(i) for i in tf.global_variables()]
        var_all1 = var_all[47:]
        # var_all1 = var_all[74:]
        # Init variables
        sess.run(tf.variables_initializer(var_all1))


        epoch = 0

        while (epoch < FLAGS.epochs_gan):
            epoch += 1
            for g_epoch in range(FLAGS.epochs_g):
                Att_dict = build_dictionary_Att(att_adj, att_fea, ori_support, att_pooling, ori_label, att_mask, Att_placeholders)
                Att_dict.update({Att_placeholders['dropout']: FLAGS.dropout})
                out_gen = sess.run([Attgraph.G_op, Attgraph.G_loss, Attgraph.gen_adj_L, Attgraph.gen_fea_L], feed_dict=Att_dict)
                # Aout_gen = sess.run([Attgraph.pre, Attgraph.accuracy_att], feed_dict=Att_dict)

                # acc_ori = sess.run([Attgraph.accuracy_ori, Attgraph.output_ori, Attgraph.class_ori], feed_dict=Att_dict)  # 原始图的预测准确率
                acc_ori = sess.run(
                    [network.accuracy_att, network.output_att, network.class_graph], feed_dict=test_dict)
                # acc_gen = sess.run([Attgraph.accuracy_att, Attgraph.pre],feed_dict=Att_dict)  # 经过G生成的目标图分类结果，验证G生成效果
                gen_support = out_gen[2]
                gen_fea = out_gen[4]
                # gen_support = preprocess_adj(coo_matrix(out_gen[2]), True, True)
                train_dict.update({GCN_placeholders['att_adj']: gen_support})
                train_dict.update({GCN_placeholders['att_fea']: gen_fea})
                acc_gen = sess.run([network.accuracy_att, network.output_att], feed_dict=train_dict)

                # print(epoch)
            for d_epoch in range(FLAGS.epochs_d):
                out_dis = sess.run([Attgraph.D_op, Attgraph.D_loss, Attgraph.D_real, Attgraph.D_fake], feed_dict=Att_dict)

            for c_epoch in range(FLAGS.epochs_c):
                Att_dict.update({Att_placeholders['labels']: att_label})
                out_cla = sess.run([Attgraph.C_op, Attgraph.C_loss, Attgraph.gen_adj_L, Attgraph.pre, Attgraph.C_op_x, Attgraph.gen_fea_L], feed_dict=Att_dict)

                # Att_dict.update({Att_placeholders['labels']: ori_label})
                # att_acc = sess.run([Attgraph.accuracy_att, Attgraph.accuracy_ori, Attgraph.pre, Attgraph.output_ori, Attgraph.class_att], feed_dict=Att_dict)
                #攻击后生成的子图
                support_C = out_cla[2]
                fea_C = out_cla[5]
                # att_support = preprocess_adj(coo_matrix(support_C), True, True)
                att_support = support_C
                train_dict.update({GCN_placeholders['att_adj']: att_support})
                train_dict.update({GCN_placeholders['att_fea']: fea_C})
                att_acc = sess.run([network.accuracy_att, network.output_att, network.class_graph], feed_dict=train_dict)
                # att_acc = sess.run([Attgraph.accuracy_att, Attgraph.accuracy_ori, Attgraph.pre, Attgraph.output_ori,
                #                     Attgraph.class_att], feed_dict=Att_dict)



            eval_c = eval_C(att_adj, support_C)
            eval_x = eval_X_onehot(att_fea_dense[:att_size], fea_C[:att_size])
            print("Epoch:", '%04d' % (epoch), "D_loss:", out_dis[1], "G_loss=", "{:.5f}".format(out_gen[1]), "C_loss=","{:.3f}".format(out_cla[1]))
            print("         Ori_acc=", acc_ori[0], "  Gen_acc=", acc_gen[0], "C_acc=", att_acc[0], "A_modify_num", eval_c,"  X_modify_num",eval_x)


            if acc_ori[2][0] != ori_class or eval_c==0:
                break
            if epoch > 1 and eval_c!=0 or eval_x!=0:

                # 当原始gcn对干净网络中目标节点的预测正确，再判断原始gcn对攻击后的网络的攻击效果
                if acc_ori[2][0] == ori_class:
                    gcn_acc = 1

                    # 当攻击后网络对原始gcn有攻击效果，并且修改连边数小于我们的预算，就停止攻击，输出结果。
                    if att_acc[0] ==0 and eval_c < att_size*att_size * budget and eval_x < att_size * budget :
                        print("第", attack_graph, "个图攻击，原类标为", ori_class,  ",  攻击后类标为：", att_acc[2][0]," 原有连边数：",graph_link,
                              "， 第", epoch, "个epoch攻击成功", "\n", "--------------****************---------------")
                        acc_node = 1
                        AML += eval_c
                        get_txt(support_C, adj_save + '_suc.txt')
                        att_success = 1
                        break
            if epoch == 200:  # 攻击超过200次无效，就失败
                # get_txt(Attack_all_adj, adj_save + '_fail.txt')
                print("第", attack_graph, "个图攻击失败。")
        print(1)
    gcn_accuracy += gcn_acc
    attack_success += att_success

    # if gcn_accuracy != 0:  # 原gcn对节点预测结果，可能出现原gcn对某个节点的预测准确率为0
    #     ASR = attack_success / gcn_accuracy
    # else:
    #     Acc_graph.append(1)
    #     print("原GCN预测准确率为0，忽视该节点")
    modify_N = eval_c

    print("第", attack_graph, "个节点攻击，图节点数：", att_size, "，连边数：", graph_link, "，平均修改数:", "{:.3f}".format(modify_N),
          ",  节点原类标为", ori_class,
          ",  GCN预测准确率：", gcn_accuracy, ",  攻击成功率：", "{:.3f}".format(acc_node))
    print("ori_suc=", gcn_accuracy, "   att_suc=", attack_success)
    f.write(str(attack_graph) + " " + str(gcn_acc) + " " + str(acc_node) + " " + str(modify_N) + " " + " 第" + str(
        attack_graph) + "个图攻击，图节点数：" + str(node_num) + "，连边数：" + str(graph_link) + "，平均修改数:" + str(
        "{:.3f}".format(modify_N)) +
            ",  节点原类标为" + str(ori_class) +
            ",  GCN预测准确率：" + str(gcn_acc) + ",  攻击成功率：" + str("{:.3f}".format(acc_node)) + "\n")
    attack_num += 1
    f.close()
    sess.close()
    tf.reset_default_graph()

ASR = attack_success / gcn_accuracy
AML = AML/attack_success
print("ASR = ", ASR,"      AML:",AML )



